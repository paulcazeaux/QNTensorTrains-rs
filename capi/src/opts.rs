// Copyright (C) 2023-2026 Paul Cazeaux
//
// This file is part of qntt. qntt is free software: you can
// redistribute it and/or modify it under the terms of the GNU
// General Public License as published by the Free Software
// Foundation, either version 3 of the License, or (at your
// option) any later version. See LICENSE for details.

//! `#[repr(C)]` plain-data mirrors of the core-crate option structs, plus `*_default` fillers so a
//! driver never has to hard-code the defaults, some of which are derived from the space.

use std::os::raw::c_int;

use qntt::solvers::als::AlsOptions;
use qntt::solvers::excited::ExcitedInfo;
use qntt::solvers::hess::{
    CurvatureReg, RpaKernel, RpaMetric, SectorMomentOpts, SectorOpts, SectorReadout,
    SectorStaticOpts, SympPrecond, SymplecticDavidsonOpts, TdaDavidsonOpts, TdaPrecond,
};
use qntt::solvers::joint::PropRoute;
use qntt::solvers::lanczos::LanczosOptions;
use qntt::solvers::mals::{MalsOptions, NoiseType};
use qntt::solvers::multistate::{SaSweep, StateAverageOptions};
use qntt::solvers::riemannian::RiemannianOpts;
use qntt::solvers::ssmals::{SsSweep, StateSpecificOptions};
use qntt::solvers::tdvp::TdvpOptions;

use crate::enums;
use crate::err::Error;

/// Mirror of [`AlsOptions`].
#[repr(C)]
#[derive(Clone, Copy)]
pub struct QnttAlsOptions {
    pub eps: f64,
    pub max_iter: usize,
    /// Nonzero prints the per-sweep trace to the process's stdout.
    pub verbose: c_int,
}

impl QnttAlsOptions {
    pub fn to_core(self) -> AlsOptions {
        AlsOptions {
            eps: self.eps,
            max_iter: self.max_iter,
            verbose: self.verbose != 0,
        }
    }

    pub fn defaults() -> Self {
        let d = AlsOptions::default();
        QnttAlsOptions {
            eps: d.eps,
            max_iter: d.max_iter,
            verbose: d.verbose as c_int,
        }
    }
}

/// Mirror of [`MalsOptions`]. The bond-dimension and noise schedules are passed alongside as
/// caller-owned arrays because they are variable-length.
#[repr(C)]
#[derive(Clone, Copy)]
pub struct QnttMalsOptions {
    pub eps: f64,
    pub max_iter: usize,
    pub verbose: c_int,
    /// Seed for the noise draws; irrelevant when the noise schedule is all-zero.
    pub seed: u64,
    /// Noise flavour: 0 = random, 1 = perturbative (White density-matrix noise, the default —
    /// [`QnttMalsOptions::defaults`] derives the code from [`MalsOptions`]'s own `Default`).
    pub noise_type: c_int,
    /// Relative energy-improvement stopping test between outer iterations. `0.0` (the default)
    /// means "use `eps`", bit-identical to the driver's historical behaviour; a positive value
    /// separates the stopping test from the truncation budget, so `eps = 1e-2` with
    /// `conv_tol = 1e-10` is a loose truncation swept to convergence.
    pub conv_tol: f64,
}

impl QnttMalsOptions {
    /// `noise` may be empty, which means "no noise" — the ABI's convenience spelling of `[0.0]`,
    /// so a caller that predates the schedule keeps working unchanged.
    pub fn to_core(self, bond_dims: &[usize], noise: &[f64]) -> Result<MalsOptions, Error> {
        if bond_dims.is_empty() {
            return Err(Error::invalid(
                "mals bond_dims must be non-empty (use [0] for uncapped)",
            ));
        }
        if noise.iter().any(|a| !a.is_finite() || *a < 0.0) {
            return Err(Error::invalid(
                "mals noise entries must be finite and non-negative",
            ));
        }
        if !self.conv_tol.is_finite() || self.conv_tol < 0.0 {
            return Err(Error::invalid(
                "mals conv_tol must be finite and non-negative (0 = use eps)",
            ));
        }
        Ok(MalsOptions {
            eps: self.eps,
            conv_tol: (self.conv_tol > 0.0).then_some(self.conv_tol),
            bond_dims: bond_dims.to_vec(),
            noise: if noise.is_empty() {
                vec![0.0]
            } else {
                noise.to_vec()
            },
            seed: self.seed,
            noise_type: enums::noise_type(self.noise_type)?,
            max_iter: self.max_iter,
            verbose: self.verbose != 0,
        })
    }

    pub fn defaults() -> Self {
        let d = MalsOptions::default();
        QnttMalsOptions {
            eps: d.eps,
            max_iter: d.max_iter,
            verbose: d.verbose as c_int,
            seed: d.seed,
            noise_type: enums::noise_type_code(d.noise_type),
            conv_tol: d.conv_tol.unwrap_or(0.0),
        }
    }
}

/// One outer iteration's diagnostics, mirroring [`qntt::solvers::mals::MalsSweep`].
#[repr(C)]
#[derive(Clone, Copy, Default)]
pub struct QnttMalsSweep {
    pub energy: f64,
    pub discarded_weight: f64,
    pub max_bond_dim: usize,
    pub bond_dim: usize,
    pub noise: f64,
}

/// Mirror of [`TdvpOptions`].
#[repr(C)]
#[derive(Clone, Copy)]
pub struct QnttTdvpOptions {
    pub dtau: f64,
    pub eps: f64,
    pub trunc: f64,
    pub max_sweeps: usize,
    /// Pooled total bond-dimension cap (SU(2): reduced multiplets); `0` = none.
    pub chi: usize,
    pub exptol: f64,
    pub verbose: c_int,
}

impl QnttTdvpOptions {
    pub fn to_core(self) -> TdvpOptions {
        TdvpOptions {
            dtau: self.dtau,
            eps: self.eps,
            trunc: self.trunc,
            max_sweeps: self.max_sweeps,
            chi: qntt::Pooled(self.chi),
            exptol: self.exptol,
            verbose: self.verbose != 0,
        }
    }

    pub fn defaults() -> Self {
        let d = TdvpOptions::default();
        QnttTdvpOptions {
            dtau: d.dtau,
            eps: d.eps,
            trunc: d.trunc,
            max_sweeps: d.max_sweeps,
            chi: d.chi.0,
            exptol: d.exptol,
            verbose: d.verbose as c_int,
        }
    }
}

/// Mirror of [`LanczosOptions`].
#[repr(C)]
#[derive(Clone, Copy)]
pub struct QnttLanczosOptions {
    pub tol: f64,
    pub max_iter: usize,
}

impl QnttLanczosOptions {
    pub fn to_core(self) -> LanczosOptions {
        LanczosOptions {
            tol: self.tol,
            max_iter: self.max_iter,
        }
    }

    pub fn defaults() -> Self {
        let d = LanczosOptions::default();
        QnttLanczosOptions {
            tol: d.tol,
            max_iter: d.max_iter,
        }
    }
}

/// Mirror of [`RiemannianOpts`].
#[repr(C)]
#[derive(Clone, Copy)]
pub struct QnttRiemannianOpts {
    pub eps: f64,
    pub max_iter: usize,
    pub gradtol: f64,
    pub seed: u64,
    /// 0 = Fletcher–Reeves, 1 = Polak–Ribière (CG only; ignored by gradient descent).
    pub variant: c_int,
}

impl QnttRiemannianOpts {
    pub fn to_core(self) -> Result<RiemannianOpts, Error> {
        Ok(RiemannianOpts {
            eps: self.eps,
            max_iter: self.max_iter,
            gradtol: self.gradtol,
            seed: self.seed,
            variant: enums::cg_variant(self.variant)?,
            preserve_grid: false,
            retraction: qntt::solvers::riemannian::Retraction::Hosvd,
        })
    }

    /// The keyword defaults of `RiemannianOpts::new` (none of them depends on the space).
    pub fn defaults() -> Self {
        QnttRiemannianOpts {
            eps: 1e-4,
            max_iter: 100,
            gradtol: 1e-6,
            seed: 1234,
            variant: 1,
        }
    }
}

/// The matrix-free RPA / linear-response options: the C face of [`SymplecticDavidsonOpts`]'s
/// basis budget and tolerance ([`Self::to_davidson`]); the other knobs take the library defaults.
#[repr(C)]
#[derive(Clone, Copy)]
pub struct QnttSymplecticLowestOpts {
    pub max_iter: usize,
    pub tol: f64,
}

impl QnttSymplecticLowestOpts {
    /// Map onto the block-Davidson options of the signed entries (`qntt_*_rpa_signed`; the call's
    /// `kernel` argument then sets [`SymplecticDavidsonOpts::kernel`]): `max_iter` becomes the
    /// budget in PAIR-EQUIVALENTS (`max_basis`: the paired kernel spends it as `max_iter` `(A, B)`
    /// pairs, the signed W-metric kernel as `⌈max_iter/2⌉` basis vectors at
    /// 2 applies each), `tol` the residual tolerance (ω units on the signed entries); block size,
    /// subspace bound, warmup and the preconditioner arm take the library defaults — the
    /// preconditioner default is `SympPrecond::Auto`, which this TT-layer entry point resolves to
    /// the fixed unshifted diagonal inverse (D0): measured to win or tie every TT cell. The
    /// θ-shifted tangent diagonal was measured to stall convergence rather than help it and was
    /// deleted (the measured tables live on the enum's variants).
    pub fn to_davidson(self) -> SymplecticDavidsonOpts {
        SymplecticDavidsonOpts {
            max_basis: self.max_iter,
            tol: self.tol,
            ..Default::default()
        }
    }

    /// The library defaults, read back through [`Self::to_davidson`]'s map: `max_iter` is
    /// [`SymplecticDavidsonOpts::max_basis`], `tol` its `tol`.
    pub fn defaults() -> Self {
        let d = SymplecticDavidsonOpts::default();
        QnttSymplecticLowestOpts {
            max_iter: d.max_basis,
            tol: d.tol,
        }
    }
}

/// Wire code for [`SectorReadout::Tda`] — `eig(A^φ)`, an ENERGY.
pub const QNTT_SECTOR_READOUT_TDA: c_int = 0;
/// Wire code for [`SectorReadout::Rpa`] — the signed `ω²` of the pair `(A^φ, B^φ)`.
pub const QNTT_SECTOR_READOUT_RPA: c_int = 1;

/// Wire code for [`RpaMetric::Ii`] (`W = H^ii`).
pub const QNTT_RPA_METRIC_II: c_int = 0;
/// Wire code for [`RpaMetric::Rr`] (`W = H^rr` — the swap).
pub const QNTT_RPA_METRIC_RR: c_int = 1;
/// Wire code for [`RpaMetric::Auto`]. As a REQUEST it is `II` on every entry, with no probe:
/// to `qntt_*_rpa_signed` the side `q = X ∓ Y` the Weingarten fraction reads under the
/// paired kernel, and also the signed kernel's lane under `QNTT_RPA_KERNEL_SIGNED`
/// ([`rpa_kernel_from_code`]); to `qntt_*_sector_rpa` the `II` lane, no probe and no retry (the
/// signed sector readout never refuses on the metric). As an ECHO, "no metric was resolved" (the
/// run returned before choosing one, or the readout has no metric).
pub const QNTT_RPA_METRIC_AUTO: c_int = 2;

/// Wire code for a resolved-or-unresolved [`RpaMetric`] — the inverse of
/// [`rpa_metric_from_code`], so the echo crosses back out through one table.
pub fn rpa_metric_code(m: RpaMetric) -> c_int {
    match m {
        RpaMetric::Ii => QNTT_RPA_METRIC_II,
        RpaMetric::Rr => QNTT_RPA_METRIC_RR,
        RpaMetric::Auto => QNTT_RPA_METRIC_AUTO,
    }
}

/// A `QNTT_RPA_METRIC_*` REQUEST code as an [`RpaMetric`] — the one decoder behind
/// [`QnttSectorOpts::to_core`] and `qntt_*_rpa_signed`'s `metric` argument. An unknown code is
/// [`crate::err::QNTT_INVALID_ARG`] naming the legal codes: an out-of-range enum from a
/// dynamically typed caller is a CALLER error, not a `QNTT_PANIC`.
pub fn rpa_metric_from_code(v: c_int) -> Result<RpaMetric, Error> {
    match v {
        QNTT_RPA_METRIC_II => Ok(RpaMetric::Ii),
        QNTT_RPA_METRIC_RR => Ok(RpaMetric::Rr),
        QNTT_RPA_METRIC_AUTO => Ok(RpaMetric::Auto),
        v => Err(Error::invalid(format!(
            "RPA metric {v} (want 0 = ii (W = H^ii), 1 = rr (the swap), 2 = auto); note this is \
             NOT the QNTT_HESS_* numbering, which puts rr first"
        ))),
    }
}

/// `qntt_*_rpa_signed`'s `kernel` request and `out_kernel` echo: the paired Davidson
/// ([`RpaKernel::Paired`], the library default). Zero, so a zero-initialized caller gets the
/// default.
pub const QNTT_RPA_KERNEL_PAIRED: c_int = 0;
/// `kernel` / `out_kernel`: the signed W-metric Davidson ([`RpaKernel::Signed`]). As an echo of a
/// `PAIRED` request it also names the fallback a solver option with no paired arm takes; no such
/// option is reachable through [`QnttSymplecticLowestOpts`], so from C the echo is the request.
pub const QNTT_RPA_KERNEL_SIGNED: c_int = 1;

/// A `(kernel, metric)` request of `qntt_*_rpa_signed` (and of [`QnttSectorOpts`]'s two fields) as
/// the [`RpaKernel`] to run and the [`RpaMetric`] the caller asked for — the one decoder behind
/// every family, run BEFORE any borrow.
/// The kernel code is decoded FIRST (a bad kernel and a bad metric together report the kernel),
/// then the metric through [`rpa_metric_from_code`]. Under `PAIRED` the metric names only the side
/// `f_W` reads; under `SIGNED` it is also the kernel's lane: `RR` → `signed(Rr)`, `II` and `AUTO` →
/// `signed(Ii)` with NO probe (C has no probe budget, so the probe stays unreachable
/// on purpose). The `C64` exports pass `QNTT_RPA_METRIC_RR`, the only lane that layer
/// has. An unknown kernel code is [`crate::err::QNTT_INVALID_ARG`] naming the legal codes.
pub fn rpa_kernel_from_code(kernel: c_int, metric: c_int) -> Result<(RpaKernel, RpaMetric), Error> {
    let signed = match kernel {
        QNTT_RPA_KERNEL_PAIRED => false,
        QNTT_RPA_KERNEL_SIGNED => true,
        v => {
            return Err(Error::invalid(format!(
                "RPA kernel {v} (want 0 = paired (the library default), 1 = signed (the W-metric \
                 Davidson))"
            )));
        }
    };
    let metric = rpa_metric_from_code(metric)?;
    let kernel = if !signed {
        RpaKernel::Paired
    } else if metric == RpaMetric::Rr {
        RpaKernel::signed(RpaMetric::Rr)
    } else {
        RpaKernel::signed(RpaMetric::Ii)
    };
    Ok((kernel, metric))
}

/// Wire code for [`CurvatureReg::Off`] (`qntt_*_rpa_signed`'s `curvature_kind`): the exact
/// Riemannian Hessian; `curvature_value` is ignored, NaN included.
pub const QNTT_CURVATURE_REG_OFF: c_int = 0;
/// Wire code for [`CurvatureReg::Hard`]: `B → P·B·P`, `P` the column projector off the right
/// Schmidt directions with Schmidt value below `curvature_value` (strict).
pub const QNTT_CURVATURE_REG_HARD: c_int = 1;
/// Wire code for [`CurvatureReg::Tikhonov`]: `B → D·B·D`, `D = V diag(s²/(s² + value²)) Vᵀ`.
pub const QNTT_CURVATURE_REG_TIKHONOV: c_int = 2;

/// A `(curvature_kind, curvature_value)` request as a [`CurvatureReg`] — the decoder behind
/// `qntt_*_rpa_signed`, run after the metric and BEFORE the borrow, so a caller error costs no
/// solve. An unknown kind is [`crate::err::QNTT_INVALID_ARG`] naming the legal codes; `HARD` /
/// `TIKHONOV` with a value that is not finite and `> 0` is `QNTT_INVALID_ARG` too — the library's
/// own panic (`with_curvature`) is pre-validated here, so no caller error crosses as `QNTT_PANIC`.
/// `OFF` ignores the value, NaN included (the `penalty_rule` precedent). The value is an absolute
/// Schmidt value; on `su2_f64` it is the RAW `σ` of `qntt_*_vec_schmidt` (not `√weight·σ`).
pub fn curvature_reg_from_code(kind: c_int, value: f64) -> Result<CurvatureReg, Error> {
    let reg = match kind {
        QNTT_CURVATURE_REG_OFF => return Ok(CurvatureReg::Off),
        QNTT_CURVATURE_REG_HARD => CurvatureReg::Hard(value),
        QNTT_CURVATURE_REG_TIKHONOV => CurvatureReg::Tikhonov(value),
        k => {
            return Err(Error::invalid(format!(
                "curvature kind {k} (want 0 = off, 1 = hard, 2 = tikhonov)"
            )));
        }
    };
    if !(value.is_finite() && value > 0.0) {
        return Err(Error::invalid(format!(
            "curvature value {value} for {reg:?}: a hard or tikhonov scale must be finite and > 0 \
             (an absolute Schmidt value; 0 = off is the only spelling of no regularization)"
        )));
    }
    Ok(reg)
}

/// Wire code for [`SympPrecond::Off`] (`qntt_*_sector_rpa`'s `precond`): no preconditioner;
/// `precond_kappa` is ignored.
pub const QNTT_SECTOR_PRECOND_OFF: c_int = 0;
/// Wire code for [`SympPrecond::DiagUnshifted`] (`D0^φ`): the channel's fixed diagonal inverse —
/// of `(D_A^φ)²` on the RPA readout, of `D_A^φ` on TDA, `D_A^φ` the shifted effective-Hamiltonian
/// diagonal minus `λ` (`SectorCtx::a_diagonal`); `precond_kappa` caps its dynamic range.
pub const QNTT_SECTOR_PRECOND_D0: c_int = 1;
/// Wire code for [`SympPrecond::Auto`]: the library's SECTOR default, which the sector layer
/// resolves (`SympPrecond::resolve_sector`) to `OFF` until a measured table promotes `D0`.
pub const QNTT_SECTOR_PRECOND_AUTO: c_int = 2;

/// A `(precond, precond_kappa)` request as a [`SympPrecond`] — the decoder behind
/// `qntt_*_sector_rpa`, run with the curvature pair BEFORE any handle borrow, so a caller error
/// costs no solve. An unknown code is [`crate::err::QNTT_INVALID_ARG`] naming the legal codes.
/// `precond_kappa` is never refused: on `D0` a value `<= 0` or non-finite (NaN included) is the
/// AUTO cap `1e3` (decoded to `kappa = 0.0`, the factory's own auto spelling, so NaN and `0` give
/// the same bits); `OFF` and `AUTO` ignore it.
pub fn sector_precond_from_code(code: c_int, kappa: f64) -> Result<SympPrecond, Error> {
    match code {
        QNTT_SECTOR_PRECOND_OFF => Ok(SympPrecond::Off),
        QNTT_SECTOR_PRECOND_D0 => Ok(SympPrecond::DiagUnshifted {
            kappa: if kappa.is_finite() && kappa > 0.0 {
                kappa
            } else {
                0.0
            },
        }),
        QNTT_SECTOR_PRECOND_AUTO => Ok(SympPrecond::Auto),
        c => Err(Error::invalid(format!(
            "sector precond {c} (want 0 = off, 1 = d0 (the fixed inverse of the channel's shifted \
             diagonal), 2 = auto (the library's sector default: off until a measured table))"
        ))),
    }
}

#[cfg(test)]
fn curvature_reg_code(reg: CurvatureReg) -> (c_int, f64) {
    match reg {
        CurvatureReg::Off => (QNTT_CURVATURE_REG_OFF, 0.0),
        CurvatureReg::Hard(v) => (QNTT_CURVATURE_REG_HARD, v),
        CurvatureReg::Tikhonov(v) => (QNTT_CURVATURE_REG_TIKHONOV, v),
    }
}

/// Wire code for [`TdaPrecond::Auto`] (`QnttTdaOpts::precond`): the library default, which
/// resolves to `DIAG` at `kappa = 10` ([`TdaPrecond::resolve`]). Zero, so a zero-initialized
/// request gets the default.
pub const QNTT_TDA_PRECOND_AUTO: c_int = 0;
/// Wire code for [`TdaPrecond::Off`]: plain residual corrections.
pub const QNTT_TDA_PRECOND_OFF: c_int = 1;
/// Wire code for [`TdaPrecond::Diag`] (D0): the fixed, θ-independent inverse of the clamped
/// `D_A`; `QnttTdaOpts::precond_kappa` caps its condition number (`<= 0` or non-finite = the auto
/// cap `1e3`).
pub const QNTT_TDA_PRECOND_DIAG: c_int = 2;
/// Wire code for [`TdaPrecond::Shifted`]: the θ-shifted Davidson correction `(D_A − θᵢ)⁻¹`.
pub const QNTT_TDA_PRECOND_SHIFTED: c_int = 3;

/// The `QNTT_TDA_PRECOND_*` code of a [`TdaPrecond`] — the `out_precond` echo of `qntt_*_tda`.
/// The echoed value is the RESOLVED arm (`TdaInfo::precond`), never `Auto` on a run; an arm this
/// table does not know (the enum is non-exhaustive) is `-1`.
pub fn tda_precond_code(p: TdaPrecond) -> c_int {
    match p {
        TdaPrecond::Auto => QNTT_TDA_PRECOND_AUTO,
        TdaPrecond::Off => QNTT_TDA_PRECOND_OFF,
        TdaPrecond::Diag { .. } => QNTT_TDA_PRECOND_DIAG,
        TdaPrecond::Shifted => QNTT_TDA_PRECOND_SHIFTED,
        _ => -1,
    }
}

/// Mirror of [`TdaDavidsonOpts`] — the in-sector Tamm–Dancoff readout `qntt_*_tda`.
#[repr(C)]
#[derive(Clone, Copy, Debug)]
pub struct QnttTdaOpts {
    /// [`TdaDavidsonOpts::tol`]: ABSOLUTE residual tolerance per root, in hartree. Finite and
    /// `>= 0`, else `QNTT_INVALID_ARG`.
    pub tol: f64,
    /// [`TdaDavidsonOpts::max_iter`]: outer block-Davidson iterations.
    pub max_iter: usize,
    /// [`TdaDavidsonOpts::max_subspace`]: thick-restart bound, `0` = the auto
    /// `max(8·(k + 2), 32)`.
    pub max_subspace: usize,
    /// [`TdaDavidsonOpts::restart_keep`]: Ritz columns kept across a restart, `0` = `2k`.
    pub restart_keep: usize,
    /// [`TdaDavidsonOpts::nwarm`]: unpreconditioned warm-up iterations.
    pub nwarm: usize,
    /// [`TdaDavidsonOpts::max_concurrent`]: most `A` images computed at once, `0` = a whole batch.
    pub max_concurrent: usize,
    /// The condition-number cap of `QNTT_TDA_PRECOND_DIAG` (`<= 0` or non-finite = the auto cap
    /// `1e3`); every other arm ignores it. Default `10`, the value `AUTO` resolves to.
    pub precond_kappa: f64,
    /// `QNTT_TDA_PRECOND_*`; an unknown code is `QNTT_INVALID_ARG`.
    pub precond: c_int,
    /// [`TdaDavidsonOpts::verbosity`]: `>= 1` prints the per-iteration line to stdout. Negative is
    /// `QNTT_INVALID_ARG`; values above 255 are clamped.
    pub verbosity: c_int,
}

impl QnttTdaOpts {
    /// Decode onto [`TdaDavidsonOpts`], refusing every value the library would panic on (a
    /// negative or non-finite `tol`) and every unknown code with [`crate::err::QNTT_INVALID_ARG`]
    /// — a caller error, decoded before any handle is borrowed. `precond_kappa` is never refused:
    /// under `DIAG` a value `<= 0` or non-finite (NaN included) decodes to `kappa = 0.0`, the
    /// library's own auto spelling, so NaN and `0` give the same bits.
    pub fn to_core(self) -> Result<TdaDavidsonOpts, Error> {
        let precond = match self.precond {
            QNTT_TDA_PRECOND_AUTO => TdaPrecond::Auto,
            QNTT_TDA_PRECOND_OFF => TdaPrecond::Off,
            QNTT_TDA_PRECOND_DIAG => TdaPrecond::Diag {
                kappa: if self.precond_kappa.is_finite() && self.precond_kappa > 0.0 {
                    self.precond_kappa
                } else {
                    0.0
                },
            },
            QNTT_TDA_PRECOND_SHIFTED => TdaPrecond::Shifted,
            c => {
                return Err(Error::invalid(format!(
                    "TDA precond {c} (want 0 = auto (the library default: diag at kappa 10), \
                     1 = off, 2 = diag (D0, capped by precond_kappa), 3 = shifted); note this is \
                     NOT the QNTT_SECTOR_PRECOND_* numbering, which puts auto last"
                )));
            }
        };
        if !(self.tol.is_finite() && self.tol >= 0.0) {
            return Err(Error::invalid(format!(
                "TDA tol {} (an absolute residual tolerance in hartree: finite and >= 0)",
                self.tol
            )));
        }
        if self.verbosity < 0 {
            return Err(Error::invalid(format!(
                "TDA verbosity {} (want >= 0)",
                self.verbosity
            )));
        }
        Ok(TdaDavidsonOpts {
            tol: self.tol,
            max_iter: self.max_iter,
            max_subspace: self.max_subspace,
            restart_keep: self.restart_keep,
            nwarm: self.nwarm,
            precond,
            max_concurrent: self.max_concurrent,
            verbosity: self.verbosity.min(c_int::from(u8::MAX)) as u8,
        })
    }

    /// The library defaults ([`TdaDavidsonOpts::default`]); `precond_kappa = 10`, the cap `AUTO`
    /// resolves to, so switching the code to `DIAG` alone reproduces the default arm.
    pub fn defaults() -> Self {
        let d = TdaDavidsonOpts::default();
        QnttTdaOpts {
            tol: d.tol,
            max_iter: d.max_iter,
            max_subspace: d.max_subspace,
            restart_keep: d.restart_keep,
            nwarm: d.nwarm,
            max_concurrent: d.max_concurrent,
            precond_kappa: match TdaPrecond::Auto.resolve() {
                TdaPrecond::Diag { kappa } => kappa,
                _ => 10.0,
            },
            precond: tda_precond_code(d.precond),
            verbosity: c_int::from(d.verbosity),
        }
    }
}

/// Mirror of [`SectorOpts`] (cross-sector / shifted-channel response).
#[repr(C)]
#[derive(Clone, Copy)]
pub struct QnttSectorOpts {
    /// `Tda`: outer block-Davidson iterations. `Rpa`: the budget in pair-equivalents for either
    /// kernel (`(A, B)` apply pairs: the paired kernel's basis vectors, or `⌈max_iter/2⌉` signed
    /// basis vectors of 2 Hessian-class applies each) — the same accounting
    /// [`QnttSymplecticLowestOpts::to_davidson`] documents.
    pub max_iter: usize,
    pub tol: f64,
    /// [`SectorOpts::max_subspace`]: the `Rpa` arm's thick-restart bound, `0` = the kernel's auto
    /// bound. Ignored by the `Tda` readout.
    pub max_subspace: usize,
    /// `QNTT_SECTOR_READOUT_*`. Selects the ARM, one readout per entry point in the library.
    pub readout: c_int,
    /// `QNTT_RPA_METRIC_*`, `qntt_*_rpa_signed`'s `metric` rule: under `kernel = PAIRED` the side
    /// `q = X ∓ Y` the Weingarten fraction reads; under `SIGNED` also the kernel's lane (`AUTO` =
    /// `II` under both, no probe). Ignored by the `Tda` readout.
    pub metric: c_int,
    /// `QNTT_RPA_KERNEL_*` ([`SectorOpts::kernel`]): `PAIRED` (`0`, the library default) or
    /// `SIGNED`, decoded by [`rpa_kernel_from_code`]. Ignored by the `Tda` readout.
    pub kernel: c_int,
    /// Explicit padding (write `0`); never read.
    pub reserved: c_int,
}

impl QnttSectorOpts {
    /// `seed` is a call parameter, not an option, so it is passed in. Returns the core options
    /// and the decoded `metric` request (the `f_W` side under the paired kernel; the lane under
    /// the signed one, where it equals the kernel's own).
    pub fn to_core(self, seed: u64) -> Result<(SectorOpts, RpaMetric), Error> {
        let readout = match self.readout {
            QNTT_SECTOR_READOUT_TDA => SectorReadout::Tda,
            QNTT_SECTOR_READOUT_RPA => SectorReadout::Rpa,
            v => {
                return Err(Error::invalid(format!(
                    "sector readout {v} (want 0 = tda (eig A^phi, an energy), 1 = rpa (signed \
                     omega^2))"
                )));
            }
        };
        let (kernel, metric) = rpa_kernel_from_code(self.kernel, self.metric)?;
        Ok((
            SectorOpts {
                tol: self.tol,
                max_iter: self.max_iter,
                readout,
                seed,
                kernel,
                precond: SympPrecond::Auto,
                max_subspace: self.max_subspace,
            },
            metric,
        ))
    }

    pub fn defaults() -> Self {
        let d = SectorOpts::default();
        QnttSectorOpts {
            max_iter: d.max_iter,
            tol: d.tol,
            max_subspace: d.max_subspace,
            readout: match d.readout {
                SectorReadout::Tda => QNTT_SECTOR_READOUT_TDA,
                SectorReadout::Rpa => QNTT_SECTOR_READOUT_RPA,
            },
            metric: QNTT_RPA_METRIC_AUTO,
            kernel: match d.kernel {
                RpaKernel::Paired => QNTT_RPA_KERNEL_PAIRED,
                RpaKernel::Signed { .. } => QNTT_RPA_KERNEL_SIGNED,
            },
            reserved: 0,
        }
    }
}

/// Wire code for [`PropRoute::Shifted`] — accepted at the wire and REFUSED with
/// [`crate::err::QNTT_INVALID_ARG`], so a caller reusing the joint layer's vocabulary gets a
/// message instead of a panic. It is the abelian joint layer's production route, which is why it
/// must be named here.
pub const QNTT_PROP_ROUTE_SHIFTED: c_int = 0;
/// Wire code for [`PropRoute::Materialized`] — the only route a channel HAS: `project_hmul_env`
/// needs `μ̂` as a `SparseHamiltonian` on `x`'s own space, which a cross-irrep property is not.
pub const QNTT_PROP_ROUTE_MATERIALIZED: c_int = 1;

fn prop_route_from_code(route: c_int) -> Result<PropRoute, Error> {
    match route {
        QNTT_PROP_ROUTE_MATERIALIZED => Ok(PropRoute::Materialized),
        QNTT_PROP_ROUTE_SHIFTED => Err(Error::invalid(
            "property route 0 (shifted): `Shifted` does not exist on a channel — \
             project_hmul_env needs the property as a SparseHamiltonian on x's OWN space, which a \
             cross-irrep property is not. Pass QNTT_PROP_ROUTE_MATERIALIZED (1), the only route a \
             channel has",
        )),
        v => Err(Error::invalid(format!(
            "property route {v} (want 1 = QNTT_PROP_ROUTE_MATERIALIZED, the only route a channel \
             has; 0 = QNTT_PROP_ROUTE_SHIFTED is named on the wire and refused, because \
             project_hmul_env needs the property as a SparseHamiltonian on x's OWN space, which a \
             cross-irrep property is not)"
        ))),
    }
}

fn positive_finite(name: &str, v: f64) -> Result<f64, Error> {
    if v.is_finite() && v > 0.0 {
        Ok(v)
    } else {
        Err(Error::invalid(format!(
            "{name} = {v:e} must be finite and strictly positive"
        )))
    }
}

fn positive_tt_grad_tol(v: f64) -> Result<f64, Error> {
    if v > 0.0 && !v.is_nan() {
        Ok(v)
    } else {
        Err(Error::invalid(format!(
            "tt_grad_tol = {v:e} must be strictly positive (+inf IS accepted: it is the \
             documented diagnosis-only escape that makes the TT-criticality gate vacuous)"
        )))
    }
}

/// Mirror of [`SectorStaticOpts`] — the knobs of `qntt_*_sector_alpha`.
#[repr(C)]
#[derive(Clone, Copy)]
pub struct QnttSectorAlphaOpts {
    /// Relative residual tolerance of each PCG solve. Default `1e-8`.
    pub tol: f64,
    /// Iteration cap per solve. Default `500`; CG terminates in at most `dim T^φ` exact steps.
    pub max_iter: usize,
    /// TT-criticality gate on `‖P((H−λ)x)‖`, RELATIVE to `max(1,|λ|)`. Default `1e-6`; `+inf` is
    /// the documented diagnosis-only escape.
    pub tt_grad_tol: f64,
    /// Hard-assert bound on the α symmetry defect, relative to `max(1, max|α|)`. Read ONLY when
    /// [`Self::has_sym_assert_rel`] is nonzero.
    pub sym_assert_rel: f64,
    /// `0` ⇒ [`SectorStaticOpts::sym_assert_rel`] `= None`, which is what makes the library
    /// recompute `100·tol` from the ACTUAL [`Self::tol`] at solve time rather than freeze it here.
    pub has_sym_assert_rel: c_int,
    /// `QNTT_PROP_ROUTE_*`. `MATERIALIZED` is the only legal value today.
    pub route: c_int,
}

impl QnttSectorAlphaOpts {
    /// `Option<f64>` on the wire is the `has_*` + value pair, the convention
    /// [`QnttGadInfo::has_index_estimate`] already ships. `has_sym_assert_rel = 0` maps to `None`,
    /// which is what makes the library recompute `100·tol` from the ACTUAL `tol` at solve time —
    /// a wire-side `100*tol` would freeze the bound and blame a healthy loosened-tol solve.
    pub fn to_static(self) -> Result<SectorStaticOpts, Error> {
        prop_route_from_code(self.route)?;
        let tol = positive_finite("tol", self.tol)?;
        if self.max_iter == 0 {
            return Err(Error::invalid(
                "max_iter = 0: a solve with no iterations publishes a bound of unknown tightness",
            ));
        }
        let sym_assert_rel = if self.has_sym_assert_rel != 0 {
            Some(positive_finite("sym_assert_rel", self.sym_assert_rel)?)
        } else {
            None
        };
        Ok(SectorStaticOpts {
            tol,
            max_iter: self.max_iter,
            tt_grad_tol: positive_tt_grad_tol(self.tt_grad_tol)?,
            sym_assert_rel,
        })
    }

    /// Built from [`SectorStaticOpts::default()`] field by field, never from retyped literals —
    /// which is what makes `sector_obs_options_default_mirrors_the_rust_defaults` a real test.
    pub fn defaults() -> Self {
        let d = SectorStaticOpts::default();
        QnttSectorAlphaOpts {
            tol: d.tol,
            max_iter: d.max_iter,
            tt_grad_tol: d.tt_grad_tol,
            sym_assert_rel: d.sym_assert_rel.unwrap_or(0.0),
            has_sym_assert_rel: c_int::from(d.sym_assert_rel.is_some()),
            route: QNTT_PROP_ROUTE_MATERIALIZED,
        }
    }
}

/// Mirror of [`SectorMomentOpts`] — the knobs of `qntt_*_sector_transition_moments`, minus
/// `solver`.
#[repr(C)]
#[derive(Clone, Copy)]
pub struct QnttSectorMomentOpts {
    /// TT-criticality gate, RELATIVE to `max(1,|λ|)`. Default `1e-6`: the moments consume `B^φ`
    /// through the Grams. `+inf` is the documented diagnosis-only escape.
    pub tt_grad_tol: f64,
    /// RELATIVE window grouping `ω_n` into degenerate clusters. Default `1e-8`. Not a tolerance
    /// on the answer — a statement about which numbers EXIST.
    pub degeneracy_tol: f64,
    /// PSD tolerance of the Williamson reduction. Default `1e-10`.
    pub psd_tol: f64,
    /// `QNTT_PROP_ROUTE_*`. `MATERIALIZED` is the only legal value today.
    pub route: c_int,
    /// Explicit tail padding, so `sizeof` equals the sum of the field sizes. Write `0`.
    pub reserved: c_int,
}

impl QnttSectorMomentOpts {
    /// `route` is validated here; `solver` stays at [`SectorOpts::default()`] and is UNREAD on the
    /// dense route, which is the only route the ABI exposes.
    pub fn to_moment(self) -> Result<SectorMomentOpts, Error> {
        let route = prop_route_from_code(self.route)?;
        Ok(SectorMomentOpts {
            tt_grad_tol: positive_tt_grad_tol(self.tt_grad_tol)?,
            degeneracy_tol: positive_finite("degeneracy_tol", self.degeneracy_tol)?,
            psd_tol: positive_finite("psd_tol", self.psd_tol)?,
            route,
            ..SectorMomentOpts::default()
        })
    }

    /// From [`SectorMomentOpts::default()`], field by field.
    pub fn defaults() -> Self {
        let d = SectorMomentOpts::default();
        QnttSectorMomentOpts {
            tt_grad_tol: d.tt_grad_tol,
            degeneracy_tol: d.degeneracy_tol,
            psd_tol: d.psd_tol,
            route: match d.route {
                PropRoute::Auto | PropRoute::Materialized => QNTT_PROP_ROUTE_MATERIALIZED,
                PropRoute::Shifted => QNTT_PROP_ROUTE_SHIFTED,
            },
            reserved: 0,
        }
    }
}

/// Scalars of `qntt_*_sector_alpha`, from `SectorStaticResponse`.
#[repr(C)]
#[derive(Clone, Copy)]
pub struct QnttSectorAlphaInfo {
    /// `max|raw[u,v] − raw[v,u]|` BEFORE symmetrization; `0` for one property.
    pub alpha_defect: f64,
    /// `min_schmidt` at the base point — quote it with every number.
    pub sigma_min: f64,
    /// `dim T^φ`, **NOT** the target sector's dimension.
    pub dim: usize,
    /// The channel's irrep flux, as a 0-based XOR index.
    pub phi: c_int,
    /// `Γ0 XOR φ`, the irrep the response reaches.
    pub target: c_int,
}

/// Scalars and SHAPE of `qntt_*_sector_transition_moments`, from `SectorTransitionMoments`.
#[repr(C)]
#[derive(Clone, Copy)]
pub struct QnttSectorMomentsInfo {
    /// `max|⟨b_i,b_j⟩ − δ_ij|` over the channel basis — MEASURED, not inherited.
    pub orth_error: f64,
    /// `min_schmidt` at the base point.
    pub sigma_min: f64,
    /// `dim T^φ`.
    pub dim: usize,
    /// REAL modes only: the length of `out_omega` and of each moment row.
    pub n_modes: usize,
    /// Imaginary modes; `> 0` forces [`Self::has_alpha_sos`] `= 0`.
    pub n_neg: usize,
    /// Degenerate `ω` clusters; `out_clusters` holds `2·n_clusters` entries.
    pub n_clusters: usize,
    /// `QNTT_RPA_METRIC_II` / `_RR`; `_AUTO` on a structurally empty channel, where no metric was
    /// resolved at all.
    pub metric: c_int,
    /// `0` ⇔ `n_neg > 0`; `out_alpha_sos` is NOT written when `0`.
    pub has_alpha_sos: c_int,
    /// `1` on the dense route: the sums are genuine, not partial.
    pub complete_basis: c_int,
    /// The channel's irrep flux, as a 0-based XOR index.
    pub phi: c_int,
    /// `Γ0 XOR φ`.
    pub target: c_int,
    /// Explicit tail padding, so `sizeof` equals the sum of the field sizes. Reads `0`.
    pub reserved: c_int,
}

/// Mirror of [`qntt::solvers::joint::cas::CasPartition`]: the `core | active | virtual` split of
/// the orbital chain, in site order. `n_core + n_active + n_virt` must equal `d`.
#[repr(C)]
#[derive(Clone, Copy)]
pub struct QnttCasPartition {
    pub n_core: usize,
    pub n_active: usize,
    pub n_virt: usize,
}

impl QnttCasPartition {
    pub fn to_core(self) -> qntt::solvers::joint::cas::CasPartition {
        qntt::solvers::joint::cas::CasPartition {
            n_core: self.n_core,
            n_active: self.n_active,
            n_virt: self.n_virt,
        }
    }
}

/// The subset of [`qntt::solvers::joint::JointScfOpts`] the C ABI exposes. Every other knob
/// (trust-region stack, Schur route, polish, …) stays at the core crate's default; the orbital
/// pair mask is derived from the [`QnttCasPartition`] inside the shim, so the only mode reachable
/// from here is the sector-pinned CAS run.
#[repr(C)]
#[derive(Clone, Copy)]
pub struct QnttCasscfOptions {
    /// Outer (orbital) iteration cap.
    pub orb_max_iter: usize,
    /// Convergence threshold on the KEPT orbital gradient norm.
    pub orb_grad: f64,
    /// TT criticality gate, relative to `max(1, |λ|)`.
    pub tt_grad_tol: f64,
    /// Keep-rule floor on the eigenvalues of the orbital metric `N`, relative to `max(1, μ_max)`.
    pub orb_floor: f64,
    /// Nonzero keeps the declared (pinned) rank grid through the TT legs. **Defaults to nonzero
    /// here** — the opposite of the core default — because the only state reachable from this
    /// entry point is a pinned CAS reference, and the pin-blind entry round would regrow it.
    pub tt_preserve_grid: c_int,
    /// Operator construction of every rebuilt Hamiltonian (`QNTT_CONSTRUCTION_*`; `0` = layer
    /// default). Ignored on the SU(2) layer, which has a single construction.
    pub construction: c_int,
    /// Nonzero prints the per-iteration trace to stdout.
    pub verbose: c_int,
}

impl QnttCasscfOptions {
    /// Fold the exposed knobs over the core defaults. `pairs` is supplied by the caller (the
    /// shim), never by the options.
    pub fn to_core(
        self,
        pairs: qntt::rotation::PairMask,
    ) -> Result<qntt::solvers::joint::JointScfOpts, Error> {
        if self.orb_grad <= 0.0
            || self.tt_grad_tol <= 0.0
            || self.orb_grad.is_nan()
            || self.tt_grad_tol.is_nan()
        {
            return Err(Error::invalid(
                "casscf: orb_grad and tt_grad_tol must be positive",
            ));
        }
        if self.orb_floor < 0.0 || self.orb_floor.is_nan() {
            return Err(Error::invalid("casscf: orb_floor must be non-negative"));
        }
        let d = qntt::solvers::joint::JointScfOpts::default();
        Ok(qntt::solvers::joint::JointScfOpts {
            orb_max_iter: self.orb_max_iter,
            orb_grad: self.orb_grad,
            orb_grad_full: Some(self.orb_grad),
            tt_grad_tol: self.tt_grad_tol,
            orb_floor: self.orb_floor,
            tt_preserve_grid: self.tt_preserve_grid != 0,
            construction: enums::construction(self.construction)?.unwrap_or(d.construction),
            verbose: self.verbose != 0,
            pairs: Some(pairs),
            ..d
        })
    }

    /// The flagship's `scf_opts` (`examples/tutorial/step_13_casscf_response.rs`), not the bare core default:
    /// 40 outer iterations, the Median construction, the pinned grid held.
    pub fn defaults() -> Self {
        let d = qntt::solvers::joint::JointScfOpts::default();
        QnttCasscfOptions {
            orb_max_iter: 40,
            orb_grad: d.orb_grad,
            tt_grad_tol: d.tt_grad_tol,
            orb_floor: d.orb_floor,
            tt_preserve_grid: 1,
            construction: 1,
            verbose: d.verbose as c_int,
        }
    }
}

/// Scalar diagnostics of a joint run — the flat part of
/// [`qntt::solvers::joint::JointScfInfo`]. The per-iteration vectors and the refusal message
/// are read from the diagnostics handle (`qntt_casscf_diag_*`).
#[repr(C)]
#[derive(Clone, Copy, Default)]
pub struct QnttCasscfInfo {
    /// Both gates met at a TT-fresh point. **Publication licence is `converged && !rank_limited`.**
    pub converged: c_int,
    pub rank_limited: c_int,
    pub stalled_at_floor: c_int,
    pub iters: usize,
    /// Kept (non-redundant) orbital coordinates at the endpoint.
    pub n_eff: usize,
    pub ls_build_count: usize,
    pub tt_legs: usize,
    pub rejections: usize,
    pub sigma_min: f64,
    pub u_orth_dev: f64,
    pub g_dropped_final: f64,
}

impl QnttCasscfInfo {
    pub fn from_core(i: &qntt::solvers::joint::JointScfInfo) -> Self {
        QnttCasscfInfo {
            converged: i.converged as c_int,
            rank_limited: i.rank_limited as c_int,
            stalled_at_floor: i.stalled_at_floor as c_int,
            iters: i.iters,
            n_eff: i.n_eff,
            ls_build_count: i.ls_build_count,
            tt_legs: i.tt_legs,
            rejections: i.rejections,
            sigma_min: i.sigma_min,
            u_orth_dev: i.u_orth_dev,
            g_dropped_final: i.g_dropped_final,
        }
    }
}

/// Mirror of [`qntt::solvers::gad::GadOpts`]. The two enum-valued knobs are flattened:
/// `w_model` + `w_damp_s` + `b_switch_tol` spell [`qntt::solvers::gad::WModel`], and
/// `floor_kind` + `floor` spell the optional `FloorPolicy`, and `rank_adapt` + `chi` +
/// `rank_eps` + `rank_period` spell the optional [`qntt::solvers::gad::RankAdaptOpts`], and
/// `border_gain_max` the [`qntt::solvers::gad::BorderPolicy`].
#[repr(C)]
#[derive(Clone, Copy)]
pub struct QnttGadOptions {
    pub gradtol: f64,
    pub max_steps: usize,
    pub alpha0: f64,
    pub alpha_min: f64,
    pub alpha_max: f64,
    pub k_davidson: usize,
    /// `QNTT_GAD_W_SWITCH` (0, the default), `QNTT_GAD_W_DAMPED` (1), `QNTT_GAD_W_FULL` (2).
    pub w_model: c_int,
    /// Under `W_DAMPED`: a positive value is the fixed Tikhonov scale; `<= 0` selects the
    /// gradient-tracked rule. Ignored otherwise.
    pub w_damp_s: f64,
    /// Under `W_SWITCH`: the `‖grad‖` threshold below which the exact `A + B` is used.
    pub b_switch_tol: f64,
    /// `QNTT_GAD_FLOOR_DEFAULT` (0: the context's own hard floor), `_HARD` (1), `_TIKHONOV` (2).
    pub floor_kind: c_int,
    /// The floor value under kinds 1 and 2.
    pub floor: f64,
    pub certify: c_int,
    pub probe_budget: usize,
    pub schmidt_floor: f64,
    pub seed: u64,
    /// Nonzero skips the pin-blind entry round — REQUIRED on a sector-pinned (CAS) grid.
    pub preserve_grid: c_int,
    /// `QNTT_GAD_RANK_FIXED` (0, the default: the declared grid), `QNTT_GAD_RANK_BOTH` (1:
    /// grow and shrink through the energy-budget retraction), `QNTT_GAD_RANK_SHRINK` (2: shrink
    /// only). A launch-zero sector is never opened under any mode, so a CAS grid stays pinned.
    pub rank_adapt: c_int,
    /// Under `RANK_BOTH`: pooled total bond-dimension cap of a growth round (SU(2): reduced
    /// multiplets; `0` = uncapped). Ignored otherwise. A non-zero value must be at least the
    /// launch point's own largest pooled total bond dimension — below it the cap truncates the
    /// start instead of adapting it, and the driver refuses it.
    pub chi: usize,
    /// Under `RANK_BOTH` / `RANK_SHRINK`: the energy budget of the adaptation round (relative,
    /// on the unit-norm state). Must be positive and finite.
    pub rank_eps: f64,
    /// Under `RANK_BOTH` / `RANK_SHRINK`: adapt every `rank_period` accepted steps (`≥ 1`).
    pub rank_period: usize,
    /// The manifold-border gain ceiling: at `gradtol`, an endpoint whose undamped Weingarten gain
    /// `‖B·V‖/‖A·V‖` exceeds it (or whose `min_schmidt` is under `schmidt_floor` with `B` live)
    /// gets no certificate — `border_verdict` set, no index, `converged` zero. Positive,
    /// finite; default `1.0`.
    pub border_gain_max: f64,
}

impl QnttGadOptions {
    pub fn to_core(self) -> Result<qntt::solvers::gad::GadOpts, Error> {
        use qntt::core::weingarten::FloorPolicy;
        use qntt::solvers::gad::WModel;
        use qntt::solvers::joint::WDampRule;
        let w_model = match self.w_model {
            0 => WModel::Switch {
                b_switch_tol: self.b_switch_tol,
            },
            1 => WModel::Damped(if self.w_damp_s > 0.0 {
                WDampRule::Fixed(self.w_damp_s)
            } else {
                WDampRule::GradTracked
            }),
            2 => WModel::Full,
            v => {
                return Err(Error::invalid(format!(
                    "gad w_model {v} (want 0=switch, 1=damped, 2=full)"
                )));
            }
        };
        let floor = match self.floor_kind {
            0 => None,
            1 => Some(FloorPolicy::Hard(self.floor)),
            2 => Some(FloorPolicy::Tikhonov(self.floor)),
            v => {
                return Err(Error::invalid(format!(
                    "gad floor_kind {v} (want 0=default, 1=hard, 2=tikhonov)"
                )));
            }
        };
        use qntt::solvers::gad::{RankAdaptMode, RankAdaptOpts};
        let rank_mode = match self.rank_adapt {
            0 => None,
            1 => Some(RankAdaptMode::Both),
            2 => Some(RankAdaptMode::ShrinkOnly),
            v => {
                return Err(Error::invalid(format!(
                    "gad rank_adapt {v} (want 0=fixed, 1=both, 2=shrink)"
                )));
            }
        };
        let rank_adapt = match rank_mode {
            None => None,
            Some(mode) => {
                if !self.rank_eps.is_finite() || self.rank_eps <= 0.0 {
                    return Err(Error::invalid(format!(
                        "gad rank_eps {} must be positive and finite",
                        self.rank_eps
                    )));
                }
                if self.rank_period == 0 {
                    return Err(Error::invalid("gad rank_period must be >= 1"));
                }
                Some(RankAdaptOpts {
                    mode,
                    chi: qntt::Pooled(self.chi),
                    round_eps: self.rank_eps,
                    period: self.rank_period,
                })
            }
        };
        if !self.border_gain_max.is_finite() || self.border_gain_max <= 0.0 {
            return Err(Error::invalid(format!(
                "gad border_gain_max {} must be positive and finite",
                self.border_gain_max
            )));
        }
        let border = qntt::solvers::gad::BorderPolicy::Refuse {
            gain_max: self.border_gain_max,
        };
        Ok(qntt::solvers::gad::GadOpts {
            gradtol: self.gradtol,
            max_steps: self.max_steps,
            alpha0: self.alpha0,
            alpha_min: self.alpha_min,
            alpha_max: self.alpha_max,
            k_davidson: self.k_davidson,
            w_model,
            floor,
            certify: self.certify != 0,
            probe_budget: self.probe_budget,
            schmidt_floor: self.schmidt_floor,
            seed: self.seed,
            preserve_grid: self.preserve_grid != 0,
            rank_adapt,
            border,
        })
    }

    pub fn defaults() -> Self {
        use qntt::solvers::gad::WModel;
        let d = qntt::solvers::gad::GadOpts::default();
        let (w_model, b_switch_tol) = match d.w_model {
            WModel::Switch { b_switch_tol } => (0, b_switch_tol),
            WModel::Damped(_) => (1, 1e-2),
            WModel::Full => (2, 1e-2),
        };
        QnttGadOptions {
            gradtol: d.gradtol,
            max_steps: d.max_steps,
            alpha0: d.alpha0,
            alpha_min: d.alpha_min,
            alpha_max: d.alpha_max,
            k_davidson: d.k_davidson,
            w_model,
            w_damp_s: 0.0,
            b_switch_tol,
            floor_kind: 0,
            floor: 1e-12,
            certify: d.certify as c_int,
            probe_budget: d.probe_budget,
            schmidt_floor: d.schmidt_floor,
            seed: d.seed,
            preserve_grid: d.preserve_grid as c_int,
            rank_adapt: 0,
            chi: qntt::solvers::gad::RankAdaptOpts::default().chi.0,
            rank_eps: qntt::solvers::gad::RankAdaptOpts::default().round_eps,
            rank_period: qntt::solvers::gad::RankAdaptOpts::default().period,
            border_gain_max: d.border.gain_max(),
        }
    }
}

/// Flat mirror of [`qntt::solvers::gad::GadInfo`]'s scalars. The two `Option`s travel as a
/// `has_*` flag plus a value that is unspecified when the flag is zero; the per-step vectors
/// (`gnorms`, `damp_s`, `trace`) are not exported.
#[repr(C)]
#[derive(Clone, Copy, Default)]
pub struct QnttGadInfo {
    pub converged: c_int,
    pub index1_certified: c_int,
    pub has_index_estimate: c_int,
    pub index_estimate: usize,
    pub steps: usize,
    pub applies: usize,
    pub gnorm: f64,
    /// ELECTRONIC Rayleigh quotient at the endpoint.
    pub lambda: f64,
    pub has_eig1: c_int,
    pub eig1: f64,
    pub v_overlap: f64,
    pub b_gain_max: f64,
    pub min_schmidt_min: f64,
    pub schmidt_floor_hit: c_int,
    pub rejected_steps: usize,
    pub relatches: usize,
    pub regime_switches: usize,
    pub radial_overlap_max: f64,
    pub branch_gap: f64,
    /// `|⟨w, V⟩|` between probe 1's witness and the tracked mode; unspecified when
    /// `has_witness_overlap` is zero (no certificate, or no negative direction found).
    pub has_witness_overlap: c_int,
    pub witness_overlap: f64,
    /// The tracker ended on the lowest branch (`witness_overlap ≥ 1 − 1e-3`). Zero with
    /// `index1_certified` set = an index-1 saddle reached on another negative-curvature branch.
    pub tracked_mode_is_lowest: c_int,
    /// Accepted steps whose base grid differed from the previous base point's (rank adaptation
    /// rounds and the fixed-rank retraction's own collapse of a deficient sector).
    pub grid_changes: usize,
    /// The endpoint reached `gradtol` on a manifold border (`min_schmidt < schmidt_floor` or the
    /// gain above `gain_max`): certificate not run, index not claimed, `converged` zero.
    pub border_verdict: c_int,
    /// Undamped `‖B·V‖/‖A·V‖` at the last refresh (NaN before any).
    pub b_gain_end: f64,
}

impl QnttGadInfo {
    pub fn from_core(i: &qntt::solvers::gad::GadInfo) -> Self {
        QnttGadInfo {
            converged: i.converged as c_int,
            index1_certified: i.index1_certified as c_int,
            has_index_estimate: i.index_estimate.is_some() as c_int,
            index_estimate: i.index_estimate.unwrap_or(0),
            steps: i.steps,
            applies: i.applies,
            gnorm: i.gnorm,
            lambda: i.lambda,
            has_eig1: i.eig1.is_some() as c_int,
            eig1: i.eig1.unwrap_or(f64::NAN),
            v_overlap: i.v_overlap,
            b_gain_max: i.b_gain_max,
            min_schmidt_min: i.min_schmidt_min,
            schmidt_floor_hit: i.schmidt_floor_hit as c_int,
            rejected_steps: i.rejected_steps,
            relatches: i.relatches,
            regime_switches: i.regime_switches,
            radial_overlap_max: i.radial_overlap_max,
            branch_gap: i.branch_gap,
            has_witness_overlap: i.witness_overlap.is_some() as c_int,
            witness_overlap: i.witness_overlap.unwrap_or(f64::NAN),
            tracked_mode_is_lowest: i.tracked_mode_is_lowest as c_int,
            grid_changes: i.grid_changes,
            border_verdict: i.border_verdict as c_int,
            b_gain_end: i.b_gain_end,
        }
    }
}

/// Mirror of [`StateAverageOptions`]. `weights`, `bond_dims` and `noise` travel alongside as
/// caller-owned arrays because they are variable-length.
#[repr(C)]
#[derive(Clone, Copy)]
pub struct QnttStateAverageOptions {
    /// The sweep schedule, read exactly as the single-state driver reads it.
    pub mals: QnttMalsOptions,
    /// Nonzero follows each root by maximum overlap instead of relabelling by energy at every
    /// bond (default 0).
    pub homing: c_int,
    /// Nonzero (the default) Löwdin-re-orthonormalizes the truncated centres at every bond.
    pub reorthonormalize: c_int,
    /// Relative size of the random perturbation added to centres `1..n` so the block solver gets
    /// `n` distinct guesses (default `0.1`). Centre 0 is never perturbed — that is what makes
    /// `nroots = 1` the single-state start bit for bit.
    pub scale: f64,
    /// Block-Davidson subspace ceiling; `0` is the ABI spelling of "the derived default".
    pub maxsubspace: usize,
    /// Block-Davidson thick-restart keep count; `0` is the ABI spelling of "the derived default".
    pub restart_keep: usize,
}

impl QnttStateAverageOptions {
    /// `weights` may be empty, meaning `1/n` each; `noise` may be empty, meaning noise-free (the
    /// `[0.0]` convenience the single-state mirror already offers). `bond_dims` must be
    /// non-empty (`[0]` = uncapped).
    pub fn to_core(
        self,
        nroots: usize,
        weights: &[f64],
        bond_dims: &[usize],
        noise: &[f64],
    ) -> Result<StateAverageOptions, Error> {
        let mals = self.mals.to_core(bond_dims, noise)?;
        if nroots == 0 {
            return Err(Error::invalid(format!(
                "state_average_mals: nroots must be ≥ 1 (got {nroots})"
            )));
        }
        if mals.noise_type == NoiseType::Random {
            return Err(Error::invalid(
                "state_average_mals: state-average sweeps define only Perturbative noise — \
                 NoiseType::Random perturbs the two-site STATE before its SVD, which has no \
                 per-state density matrix to weight and is not variational",
            ));
        }
        if !weights.is_empty() {
            validate_weights(weights, nroots, "state_average_mals")?;
        }
        if self.maxsubspace != 0 && nroots >= 2 && self.maxsubspace < 2 * nroots {
            return Err(Error::invalid(format!(
                "sa_restart_keep: maxsubspace = {} < 2·nroots = {} (the block solver needs room \
                 for the {nroots} Ritz vectors and their corrections)",
                self.maxsubspace,
                2 * nroots
            )));
        }
        Ok(StateAverageOptions {
            mals,
            weights: (!weights.is_empty()).then(|| weights.to_vec()),
            homing: self.homing != 0,
            reorthonormalize: self.reorthonormalize != 0,
            scale: self.scale,
            maxsubspace: (self.maxsubspace != 0).then_some(self.maxsubspace),
            restart_keep: (self.restart_keep != 0).then_some(self.restart_keep),
        })
    }

    pub fn defaults() -> Self {
        let d = StateAverageOptions::default();
        QnttStateAverageOptions {
            mals: QnttMalsOptions::defaults(),
            homing: d.homing as c_int,
            reorthonormalize: d.reorthonormalize as c_int,
            scale: d.scale,
            maxsubspace: d.maxsubspace.unwrap_or(0),
            restart_keep: d.restart_keep.unwrap_or(0),
        }
    }
}

fn validate_weights(w: &[f64], n: usize, who: &str) -> Result<(), Error> {
    if w.len() != n {
        return Err(Error::invalid(format!(
            "{who}: {} weights for {n} states — one per root, in label order",
            w.len()
        )));
    }
    if !w.iter().all(|a| a.is_finite() && *a > 0.0) {
        return Err(Error::invalid(format!(
            "{who}: weights must be finite and strictly positive (got {w:?}) — a zero-weight \
             state is truncated away with no trace in the ledger"
        )));
    }
    if !w.windows(2).all(|p| p[0] >= p[1]) {
        return Err(Error::invalid(format!(
            "{who}: weights must be non-increasing in label order (got {w:?})"
        )));
    }
    let sum: f64 = w.iter().sum();
    if (sum - 1.0).abs() > 1e-12 {
        return Err(Error::invalid(format!(
            "{who}: weights must sum to 1 (got {sum}) — the pooled chop's tolerance is ABSOLUTE \
             and `Σwᵢ‖Bᵢ‖²` is the trace of ρ"
        )));
    }
    Ok(())
}

/// The `Copy` part of one state-average outer iteration ([`SaSweep`]).
#[repr(C)]
#[derive(Clone, Copy, Default)]
pub struct QnttSaSweep {
    /// `Σᵢ wᵢ λᵢ` — the quantity the sweep actually minimizes.
    pub energy: f64,
    /// `maxᵢ |λᵢ − λᵢ_prev| / |λᵢ|`: the tell a single averaged energy hides when one root is
    /// still moving.
    pub max_dev: f64,
    /// How far the truncated roots drifted from orthogonality before the Löwdin step.
    pub overlap_dev: f64,
    /// Largest POOLED discarded weight over this iteration's bonds. On SU(2) at `2S > 0` this is
    /// NOT the same quantity at `nroots = 1` (the single-state kernel's unweighted θ-Frobenius
    /// ratio) as at `nroots ≥ 2` (the centerweighted projection loss).
    pub discarded_weight: f64,
    pub max_bond_dim: usize,
    pub bond_dim: usize,
    pub noise: f64,
    pub numops: usize,
}

impl QnttSaSweep {
    pub fn from_core(s: &SaSweep) -> Self {
        QnttSaSweep {
            energy: s.energy,
            max_dev: s.max_dev,
            overlap_dev: s.overlap_dev,
            discarded_weight: s.discarded_weight,
            max_bond_dim: s.max_bond_dim,
            bond_dim: s.bond_dim,
            noise: s.noise,
            numops: s.numops,
        }
    }
}

/// Mirror of [`StateSpecificOptions`]. `bond_dims`, `noise` and `lower_energies` travel alongside.
#[repr(C)]
#[derive(Clone, Copy)]
pub struct QnttStateSpecificOptions {
    /// The sweep schedule. Unlike the state average, this path ACCEPTS `noise_type = 0`
    /// (`Random`): its truncation is single-state.
    pub mals: QnttMalsOptions,
    /// [`crate::enums::ss_mode`]: 0 = penalty (the default), 1 = project.
    pub mode: c_int,
    /// [`crate::enums::penalty_rule`] kind: 0 = auto (the default), 1 = fixed.
    pub penalty_rule: c_int,
    /// The weight under `penalty_rule = 1`; ignored under 0. `0.0` is legal and means the
    /// UNPENALIZED sweep, which converges back onto the lower states — set it deliberately.
    pub penalty_value: f64,
    /// How many local roots to solve for. `1` labels by energy; `> 1` keeps the root that
    /// continues the incoming block (greedy homing).
    pub roots: usize,
    /// Block-Davidson subspace ceiling; `0` = the derived default.
    pub maxsubspace: usize,
    /// Block-Davidson thick-restart keep count; `0` = the derived default.
    pub restart_keep: usize,
}

impl QnttStateSpecificOptions {
    /// `lower_energies` may be empty, meaning "compute `RQ(ψⱼ)/‖ψⱼ‖²` once at entry"; otherwise
    /// it must hold exactly `nlower` entries.
    pub fn to_core(
        self,
        nlower: usize,
        lower_energies: &[f64],
        bond_dims: &[usize],
        noise: &[f64],
    ) -> Result<StateSpecificOptions, Error> {
        let mals = self.mals.to_core(bond_dims, noise)?;
        let mode = enums::ss_mode(self.mode)?;
        let penalty = enums::penalty_rule(self.penalty_rule, self.penalty_value)?;
        if self.roots == 0 {
            return Err(Error::invalid(format!(
                "state_specific_mals: roots = {} must be ≥ 1",
                self.roots
            )));
        }
        if !lower_energies.is_empty() && lower_energies.len() != nlower {
            return Err(Error::invalid(format!(
                "state_specific_mals: {} lower_energies for {nlower} lower state(s)",
                lower_energies.len()
            )));
        }
        if self.maxsubspace != 0 && self.roots >= 2 && self.maxsubspace < 2 * self.roots {
            return Err(Error::invalid(format!(
                "sa_restart_keep: maxsubspace = {} < 2·nroots = {} (the block solver needs room \
                 for the {} Ritz vectors and their corrections)",
                self.maxsubspace,
                2 * self.roots,
                self.roots
            )));
        }
        Ok(StateSpecificOptions {
            mals,
            mode,
            penalty,
            roots: self.roots,
            lower_energies: (!lower_energies.is_empty()).then(|| lower_energies.to_vec()),
            maxsubspace: (self.maxsubspace != 0).then_some(self.maxsubspace),
            restart_keep: (self.restart_keep != 0).then_some(self.restart_keep),
        })
    }

    pub fn defaults() -> Self {
        let d = StateSpecificOptions::default();
        let (rule, value) = enums::penalty_rule_code(d.penalty);
        QnttStateSpecificOptions {
            mals: QnttMalsOptions::defaults(),
            mode: enums::ss_mode_code(d.mode),
            penalty_rule: rule,
            penalty_value: value,
            roots: d.roots,
            maxsubspace: d.maxsubspace.unwrap_or(0),
            restart_keep: d.restart_keep.unwrap_or(0),
        }
    }
}

/// The `Copy` part of one state-specific outer iteration ([`SsSweep`]).
#[repr(C)]
#[derive(Clone, Copy, Default)]
pub struct QnttSsSweep {
    /// The PLAIN energy `λ` — the Rayleigh quotient of `H` alone, and what the caller reports.
    pub energy: f64,
    /// The PENALIZED Ritz value `θ` the sweep actually minimizes; `== energy` at `w = 0` and in
    /// project mode.
    pub penalized: f64,
    pub discarded_weight: f64,
    pub max_bond_dim: usize,
    pub bond_dim: usize,
    pub noise: f64,
    /// The weight in force this iteration; `0.0` in project mode and on the delegate path.
    pub penalty: f64,
    /// Bonds where the homed continuation was not the lowest Ritz vector (`roots > 1` only).
    pub swaps: usize,
    /// Local operator applies over both sweeps. `0` on the delegate path (`nlower == 0` with
    /// `roots == 1`), which is a documented ledger convention, not "no work done".
    pub numops: usize,
    /// Bonds whose local space could not host the requested vectors.
    pub skipped: usize,
    /// Nonzero when the stop test fired on this iteration.
    pub converged: c_int,
}

impl QnttSsSweep {
    pub fn from_core(s: &SsSweep) -> Self {
        QnttSsSweep {
            energy: s.energy,
            penalized: s.penalized,
            discarded_weight: s.discarded_weight,
            max_bond_dim: s.max_bond_dim,
            bond_dim: s.bond_dim,
            noise: s.noise,
            penalty: s.penalty,
            swaps: s.swaps,
            numops: s.numops,
            skipped: s.skipped,
            converged: s.converged as c_int,
        }
    }
}

/// Mirror of the excited-state ladder's `ExcitedOpts`, minus the four schedules (which travel
/// alongside) and minus the Rust-only `cg.preserve_grid`.
#[repr(C)]
#[derive(Clone, Copy)]
pub struct QnttExcitedOptions {
    /// How many roots the ladder returns, counting the ground state. `≥ 1`.
    pub nroots: usize,
    /// Pooled total bond-dimension cap of the state-specific sweeps and the polish manifold;
    /// `0` = full rank. On SU(2) this counts REDUCED MULTIPLET ranks.
    pub rank: usize,
    /// Truncation tolerance of every sweep, and the absolute scale of the swap scan (`10·eps`).
    pub eps: f64,
    /// Outer-iteration budget of every sweep.
    pub max_iter: usize,
    /// [`crate::enums::noise_type`]. `0` (`Random`) is refused: the state-average truncation
    /// defines only the perturbative density-matrix noise.
    pub noise_type: c_int,
    /// Seed of every stage's noise stream and of the state-average start perturbation.
    pub seed: u64,
    /// [`crate::enums::ss_mode`] for the state-specific sweeps.
    pub mode: c_int,
    /// [`crate::enums::penalty_rule`] kind.
    pub penalty_rule: c_int,
    /// The weight under `penalty_rule = 1`; it also fixes the POLISH weight.
    pub penalty_value: f64,
    /// [`crate::enums::ss_start`] (`0`; `1` is reserved).
    pub ss_start: c_int,
    /// Nonzero (the default) runs the penalized Riemannian polish after each sweep. `0` returns
    /// the sweep's own state and records `gradnorm = NaN`, `stalled = 0`, `passes = 0`.
    pub polish: c_int,
    /// Warn when a supplied lower state's exact gate exceeds this.
    pub lower_grad_tol: f64,
    /// Rank-trim tolerance (the CENTERWEIGHTED smallest Schmidt value on SU(2)).
    pub trim_tol: f64,
    /// Trim budget per root.
    pub trim_passes: usize,
    /// The polish CG's truncation tolerance. `1e-13` — the response ground-state pipeline's
    /// override, NOT `RiemannianOpts::new`'s own `1e-4`, which would stall every root far above
    /// `polish_tol`.
    pub cg_eps: f64,
    pub cg_max_iter: usize,
    /// Inert on this code path (the gradient is the exact environment contraction); carried for
    /// API stability.
    pub cg_gradtol: f64,
    pub cg_seed: u64,
    /// [`crate::enums::cg_variant`].
    pub cg_variant: c_int,
    /// Extra CG calls per polish while the exact gate is above `polish_tol`.
    pub cg_restarts: usize,
    /// The exact-gate target the restart loop drives at.
    pub polish_tol: f64,
    /// The exact gate above which a root is `stalled` (and the licence refused).
    pub grad_gate: f64,
    /// The largest off-diagonal Gram entry the licence tolerates. At a BINDING rank cap the
    /// fixed-rank penalized minimizer keeps an `O(1/w)` residual overlap and this gate is
    /// unreachable — raise `rank` or the penalty, never this.
    pub overlap_tol: f64,
    /// Nonzero also returns the state-average ladder through `out_sa_states`.
    pub keep_sa: c_int,
    pub verbose: c_int,
}

impl QnttExcitedOptions {
    /// The ladder's keyword defaults, DERIVED from `ExcitedOpts::new`.
    pub fn defaults() -> Self {
        let d = qntt::solvers::excited::ExcitedOpts::new(&qntt::QnSpace::new(2, 1, 1));
        let (rule, value) = enums::penalty_rule_code(d.penalty);
        QnttExcitedOptions {
            nroots: d.nroots,
            rank: d.rank,
            eps: d.eps,
            max_iter: d.max_iter,
            noise_type: enums::noise_type_code(d.noise_type),
            seed: d.seed,
            mode: enums::ss_mode_code(d.mode),
            penalty_rule: rule,
            penalty_value: value,
            ss_start: enums::ss_start_code(d.ss_start),
            polish: d.polish as c_int,
            lower_grad_tol: d.lower_grad_tol,
            trim_tol: d.trim_tol,
            trim_passes: d.trim_passes,
            cg_eps: d.cg.eps,
            cg_max_iter: d.cg.max_iter,
            cg_gradtol: d.cg.gradtol,
            cg_seed: d.cg.seed,
            cg_variant: enums::cg_variant_code(d.cg.variant),
            cg_restarts: d.cg_restarts,
            polish_tol: d.polish_tol,
            grad_gate: d.grad_gate,
            overlap_tol: d.overlap_tol,
            keep_sa: d.keep_sa as c_int,
            verbose: d.verbose as c_int,
        }
    }
}

/// The excited-state ladder's four variable-length schedules, borrowed from the caller.
pub struct ExcitedSchedules<'a> {
    pub weights: &'a [f64],
    pub bond_dims: &'a [usize],
    pub noise: &'a [f64],
    pub ss_noise: &'a [f64],
}

macro_rules! impl_excited_bridge {
    ($fn_name:ident, $Opts:path, $variant:path, $who:literal) => {
        /// Overwrite `base` (which must be `ExcitedOpts::new(&space)`) with the C mirror and the
        /// four schedules, re-spelling the driver's own preconditions as `QNTT_INVALID_ARG`.
        pub fn $fn_name(
            c: &QnttExcitedOptions,
            s: &ExcitedSchedules<'_>,
            mut base: $Opts,
        ) -> Result<$Opts, Error> {
            if c.nroots == 0 {
                return Err(Error::invalid(format!(
                    "excited_states: `nroots` must be at least 1 (got {}) — a ladder with no \
                     rungs has nothing to return",
                    c.nroots
                )));
            }
            let noise_type = enums::noise_type(c.noise_type)?;
            if noise_type == NoiseType::Random {
                return Err(Error::invalid(
                    "excited_states: state-average sweeps define only Perturbative noise (the \
                     pooled density matrix is what the weighted chop truncates); pass \
                     NoiseType::Perturbative",
                ));
            }
            for (what, sched) in [("noise", s.noise), ("ss_noise", s.ss_noise)] {
                if sched.iter().any(|a| !a.is_finite() || *a < 0.0) {
                    return Err(Error::invalid(format!(
                        "excited_states: `{what}` entries must be finite and non-negative"
                    )));
                }
            }
            if !s.weights.is_empty() {
                validate_weights(s.weights, c.nroots, $who)?;
            }
            base.nroots = c.nroots;
            base.rank = c.rank;
            base.eps = c.eps;
            base.max_iter = c.max_iter;
            base.noise_type = noise_type;
            base.seed = c.seed;
            base.mode = enums::ss_mode(c.mode)?;
            base.penalty = enums::penalty_rule(c.penalty_rule, c.penalty_value)?;
            base.ss_start = enums::ss_start(c.ss_start)?;
            base.polish = c.polish != 0;
            base.lower_grad_tol = c.lower_grad_tol;
            base.trim_tol = c.trim_tol;
            base.trim_passes = c.trim_passes;
            base.cg.eps = c.cg_eps;
            base.cg.max_iter = c.cg_max_iter;
            base.cg.gradtol = c.cg_gradtol;
            base.cg.seed = c.cg_seed;
            base.cg.variant = $variant(c.cg_variant)?;
            base.cg_restarts = c.cg_restarts;
            base.polish_tol = c.polish_tol;
            base.grad_gate = c.grad_gate;
            base.overlap_tol = c.overlap_tol;
            base.keep_sa = c.keep_sa != 0;
            base.verbose = c.verbose != 0;
            if !s.weights.is_empty() {
                base.weights = Some(s.weights.to_vec());
            }
            if !s.bond_dims.is_empty() {
                base.bond_dims = Some(s.bond_dims.to_vec());
            }
            if !s.noise.is_empty() {
                base.noise = s.noise.to_vec();
            }
            if !s.ss_noise.is_empty() {
                base.ss_noise = s.ss_noise.to_vec();
            }
            Ok(base)
        }
    };
}

impl_excited_bridge!(
    excited_opts_abelian,
    qntt::solvers::excited::ExcitedOpts,
    enums::cg_variant,
    "excited_states"
);
impl_excited_bridge!(
    excited_opts_su2,
    qntt::su2::solvers::excited::ExcitedOpts,
    enums::su2_cg_variant,
    "excited_states"
);

/// The scalars of the ladder's `ExcitedInfo`.
#[repr(C)]
#[derive(Clone, Copy, Default)]
pub struct QnttExcitedInfo {
    /// The largest off-diagonal entry of the returned Gram (`0.0` for a one-state ladder).
    pub overlap_dev: f64,
    /// How many rungs of `Λ = [sort(E_lower); e]` sit below the rung under them by more than
    /// `10·eps`.
    pub swaps: usize,
    /// Nonzero when stage 0's enrichment iteration ran (a rank-1 determinant/CSF start).
    pub enriched: c_int,
    /// `all(converged) ∧ overlap_dev ≤ overlap_tol ∧ swaps == 0`. It certifies convergence,
    /// orthogonality and ordering — NEVER completeness: a licensed ladder can skip a level.
    pub licence: c_int,
}

impl QnttExcitedInfo {
    pub fn from_core<V>(i: &ExcitedInfo<V>) -> Self {
        QnttExcitedInfo {
            overlap_dev: i.overlap_dev,
            swaps: i.swaps,
            enriched: i.enriched as c_int,
            licence: i.licence as c_int,
        }
    }
}

/// One row of the ladder's per-root record.
#[repr(C)]
#[derive(Clone, Copy, Default)]
pub struct QnttExcitedRoot {
    /// The state-average stage's energy for this root (`sa_history.lambdas[i]`).
    pub sa_energy: f64,
    /// The state-specific sweep's own final energy, BEFORE the polish.
    pub ss_energy: f64,
    /// The penalized functional `J` at the state-specific state.
    pub j_ss: f64,
    /// `J` at the returned (polished, trimmed) root, at the same weight.
    pub j_pol: f64,
    /// The exact penalized gate at the returned root; `NaN` when `polish = 0` (there is no
    /// critical point to certify).
    pub gradnorm: f64,
    /// The penalty weight this root's polish ran at (`0.0` when it had no lower states).
    pub penalty: f64,
    /// Trim passes performed, `0..=trim_passes`.
    pub passes: usize,
    /// Nonzero when `gradnorm > grad_gate`.
    pub stalled: c_int,
    /// Nonzero when the sweep converged and the root did not stall.
    pub converged: c_int,
    /// Nonzero when this root's rung was flagged out of order by the swap scan. Lossless as a
    /// per-root flag: `Λ`'s first `nlower` entries are sorted, so no lower rung can ever be
    /// flagged.
    pub swapped: c_int,
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::err::QNTT_INVALID_ARG;
    use qntt::solvers::joint::PropRoute;

    #[test]
    fn curvature_reg_code_inverts_the_decoder() {
        for reg in [
            CurvatureReg::Off,
            CurvatureReg::Hard(1.5e-3),
            CurvatureReg::Tikhonov(3e-4),
        ] {
            let (kind, value) = curvature_reg_code(reg);
            assert_eq!(curvature_reg_from_code(kind, value).ok(), Some(reg));
        }
    }

    #[test]
    fn to_core_maps_the_route_and_refuses_shifted_by_name() {
        assert_eq!(
            (QNTT_PROP_ROUTE_SHIFTED, QNTT_PROP_ROUTE_MATERIALIZED),
            (0, 1)
        );
        assert_eq!(PropRoute::default(), PropRoute::Auto);

        let a = QnttSectorAlphaOpts::defaults();
        assert_eq!(a.route, QNTT_PROP_ROUTE_MATERIALIZED);
        a.to_static()
            .expect("the default route is the only route a channel has");

        for route in [QNTT_PROP_ROUTE_SHIFTED, 2, -1, 7] {
            let bad = QnttSectorAlphaOpts { route, ..a };
            let e = bad
                .to_static()
                .expect_err(&format!("route {route} must be refused"));
            assert_eq!(e.code, QNTT_INVALID_ARG);
            assert!(!e.msg.is_empty(), "route {route} refused without a message");
            let m = QnttSectorMomentOpts {
                route,
                ..QnttSectorMomentOpts::defaults()
            };
            let e2 = m
                .to_moment()
                .expect_err(&format!("route {route} must be refused on both entries"));
            assert_eq!(e2.code, QNTT_INVALID_ARG);
            assert_eq!(e2.msg, e.msg, "one spelling of the refusal, not two");
        }
        let e = QnttSectorAlphaOpts {
            route: QNTT_PROP_ROUTE_SHIFTED,
            ..a
        }
        .to_static()
        .expect_err("Shifted does not exist on a channel");
        assert!(e.msg.contains("MATERIALIZED"), "{:?}", e.msg);
    }

    #[test]
    fn to_static_maps_the_optional_symmetry_bound() {
        let a = QnttSectorAlphaOpts::defaults();
        assert_eq!(a.has_sym_assert_rel, 0);
        assert_eq!(a.to_static().unwrap().sym_assert_rel, None);

        let some = QnttSectorAlphaOpts {
            has_sym_assert_rel: 1,
            sym_assert_rel: 3e-5,
            ..a
        };
        assert_eq!(some.to_static().unwrap().sym_assert_rel, Some(3e-5));
        for v in [0.0, -1.0, f64::NAN, f64::INFINITY] {
            let bad = QnttSectorAlphaOpts {
                has_sym_assert_rel: 1,
                sym_assert_rel: v,
                ..a
            };
            assert_eq!(
                bad.to_static()
                    .expect_err("sym_assert_rel must be positive and finite")
                    .code,
                QNTT_INVALID_ARG,
                "sym_assert_rel = {v}"
            );
        }
        let ignored = QnttSectorAlphaOpts {
            has_sym_assert_rel: 0,
            sym_assert_rel: f64::NAN,
            ..a
        };
        assert_eq!(ignored.to_static().unwrap().sym_assert_rel, None);
    }

    #[test]
    fn to_core_range_checks_every_knob_except_the_criticality_escape() {
        let a = QnttSectorAlphaOpts::defaults();
        for bad in [
            QnttSectorAlphaOpts { tol: 0.0, ..a },
            QnttSectorAlphaOpts { tol: -1e-8, ..a },
            QnttSectorAlphaOpts { tol: f64::NAN, ..a },
            QnttSectorAlphaOpts {
                tol: f64::INFINITY,
                ..a
            },
            QnttSectorAlphaOpts { max_iter: 0, ..a },
            QnttSectorAlphaOpts {
                tt_grad_tol: 0.0,
                ..a
            },
            QnttSectorAlphaOpts {
                tt_grad_tol: -1.0,
                ..a
            },
            QnttSectorAlphaOpts {
                tt_grad_tol: f64::NAN,
                ..a
            },
        ] {
            let e = bad
                .to_static()
                .expect_err("an out-of-range knob is QNTT_INVALID_ARG");
            assert_eq!(e.code, QNTT_INVALID_ARG);
            assert!(!e.msg.is_empty());
        }
        assert_eq!(
            QnttSectorAlphaOpts {
                tt_grad_tol: f64::INFINITY,
                ..a
            }
            .to_static()
            .expect("+inf is the documented diagnosis-only escape")
            .tt_grad_tol,
            f64::INFINITY
        );

        let m = QnttSectorMomentOpts::defaults();
        for bad in [
            QnttSectorMomentOpts {
                degeneracy_tol: 0.0,
                ..m
            },
            QnttSectorMomentOpts {
                degeneracy_tol: f64::NAN,
                ..m
            },
            QnttSectorMomentOpts { psd_tol: -1.0, ..m },
            QnttSectorMomentOpts {
                psd_tol: f64::INFINITY,
                ..m
            },
            QnttSectorMomentOpts {
                tt_grad_tol: 0.0,
                ..m
            },
        ] {
            assert_eq!(
                bad.to_moment()
                    .expect_err("an out-of-range knob is QNTT_INVALID_ARG")
                    .code,
                QNTT_INVALID_ARG
            );
        }
        assert_eq!(
            QnttSectorMomentOpts {
                tt_grad_tol: f64::INFINITY,
                ..m
            }
            .to_moment()
            .expect("+inf is the escape on both entries")
            .tt_grad_tol,
            f64::INFINITY
        );
    }

    #[test]
    fn defaults_come_from_the_library_not_from_literals() {
        let a = QnttSectorAlphaOpts::defaults().to_static().unwrap();
        let sa = qntt::solvers::hess::SectorStaticOpts::default();
        assert_eq!(
            (a.tol, a.max_iter, a.tt_grad_tol),
            (sa.tol, sa.max_iter, sa.tt_grad_tol)
        );
        assert_eq!(a.sym_assert_rel, sa.sym_assert_rel);

        let m = QnttSectorMomentOpts::defaults().to_moment().unwrap();
        let sm = qntt::solvers::hess::SectorMomentOpts::default();
        assert_eq!(
            (m.tt_grad_tol, m.degeneracy_tol, m.psd_tol),
            (sm.tt_grad_tol, sm.degeneracy_tol, sm.psd_tol)
        );
        assert_eq!(m.route, PropRoute::Materialized);
    }
}
