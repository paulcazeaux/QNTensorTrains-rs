// Copyright (C) 2023-2026 Paul Cazeaux
//
// This file is part of qntt. qntt is free software: you can
// redistribute it and/or modify it under the terms of the GNU
// General Public License as published by the Free Software
// Foundation, either version 3 of the License, or (at your
// option) any later version. See LICENSE for details.

use crate::core::SparseCore;
use crate::core::factor::Sketch;
use crate::core::tangent::{
    TtTangent, min_schmidt, shifted_channel_is_empty, shifted_frames, tangent_basis_shifted,
    tangent_dim_grids,
};
use crate::core::weingarten::FusedTermA;
use crate::hamiltonian::SparseHamiltonian;
use crate::hamiltonian::matvec::{OneSiteWorkspace, onesite_workspace};
use crate::pointgroup::Irrep;
use crate::qn::{QnSpace, RankGrid};
use crate::solvers::InnerProductVector;
use crate::solvers::als::{left_to_right_sketching, right_to_left_sketching};
use crate::solvers::davidson_block::{BlockDavidsonOpts, davidson_block_projected};
use crate::vector::{TtVector, tt_randn};

use super::davidson::{REFUSE_TAU, RpaKernel, SympPrecond, SymplecticDavidsonOpts};
use super::davidson_signed::omega_davidson_signed;
use super::lanczos::certify_negatives;
use super::mode_check::{ModeCheck, ModeCheckOps, ModeCheckOpts, TanOp, mode_checks_core};
use super::{
    NegativeWitness, PairedHook, PairedPrecond, RpaAmplitudes, RpaMetric, RpaMode, RpaModeKind,
    RpaReadout, SymplecticSpectrum, WeingartenCtx, build_symp_precond, effham_apply_shifted,
    effham_gram_shifted, omega_davidson_paired, paired_image, paired_mu, paired_precond,
    paired_route, readout_from_paired, readout_from_signed, symplectic_from_grams,
};

use rand::SeedableRng;
use rand_xoshiro::Xoshiro256PlusPlus;
use std::sync::Arc;

/// Which spectral quantity a sector readout reports. **One readout per entry point**:
/// [`SectorCtx::sector_lowest_tda`] answers only [`Self::Tda`] and
/// [`SectorCtx::sector_lowest`] only [`Self::Rpa`]; each refuses the other rather than return a
/// number in the wrong units under the caller's label.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum SectorReadout {
    /// `eig(A^φ)` — the A-only, TDA-like channel excitation ENERGIES.
    Tda,
    /// The symplectic (RPA) values of the pair `(A^φ, B^φ)`, reported as signed `ω²`.
    Rpa,
}

/// Tunables of the iterative sector readout. `k` is NOT here — it is a parameter of
/// [`SectorCtx::sector_lowest_tda`] / [`SectorCtx::sector_lowest`], because the same bound
/// operator is normally asked for different numbers of roots per channel.
#[derive(Clone, Debug)]
pub struct SectorOpts {
    /// Absolute (projected) residual tolerance per root, passed straight to the block Davidson
    /// (`Tda`) or, as [`super::SymplecticDavidsonOpts::tol`], to the symplectic solve's relative
    /// residual test (`Rpa`, ω units on both kernels).
    pub tol: f64,
    /// Iteration budget: outer iterations of the block Davidson on the `Tda` arm; on the `Rpa` arm
    /// the budget in PAIR-EQUIVALENTS, passed through as
    /// [`super::SymplecticDavidsonOpts::max_basis`] for either kernel (the paired kernel builds up
    /// to `max_iter` basis vectors at one `(A, B)` pair each; the signed W-metric kernel accepts
    /// `⌈max_iter/2⌉`, two applies each). The default 300 is shared by both arms.
    pub max_iter: usize,
    /// Which quantity to report; the entry points refuse each other's arm.
    pub readout: SectorReadout,
    /// Seed of the start block (`min(k, dim) + 2` projected random tangents, minus any the
    /// projector annihilates) — the readout is deterministic given this seed.
    pub seed: u64,
    /// The `Rpa` arm's kernel. [`RpaKernel::Paired`] (the default, as on the signed entry
    /// points): the paired-basis Davidson on `(A^φ, B^φ)`, no Hessian block as a metric.
    /// [`RpaKernel::Signed`]: the signed W-metric Davidson on the lane its `metric` names — the
    /// `Rr` lane (`W = H^rr`) for [`RpaMetric::Rr`], the `Ii` lane (`W = H^ii`) for
    /// [`RpaMetric::Ii`] and [`RpaMetric::Auto`] — ONE solve, no PD probe, no retry (the signed
    /// kernel never refuses on the metric, so a retry buys nothing). The sector
    /// layer never probes, so a `Signed` request with `probe_budget > 0` PANICS by name
    /// rather than ignore the field. The kernel that ran (and the lane) is reported in
    /// [`SectorSpectrum::Rpa`]. Ignored by the `Tda` arm.
    pub kernel: RpaKernel,
    /// The preconditioner of BOTH readouts: [`SympPrecond::Off`], [`SympPrecond::DiagUnshifted`]
    /// (`D0^φ`, the fixed inverse of the channel's shifted diagonal [`SectorCtx::a_diagonal`] —
    /// of `(D_A^φ)²` on the `Rpa` arm, of `D_A^φ` on `Tda`; θ-independent on both) or
    /// [`SympPrecond::Auto`], the default, which the sector layer resolves to `Off`
    /// ([`SympPrecond::resolve_sector`]) until a measured table promotes `D0^φ`. Every
    /// other arm is refused by name. The SU(2) sector layer takes the same three
    /// (rpa-floor-selection P4): `DiagUnshifted` is `D0^{S'}`, the fixed inverse of
    /// `Su2SectorCtx::a_diagonal` (squared on `Rpa`), behind the same guard, and `Auto` resolves
    /// to `Off` there too.
    pub precond: SympPrecond,
    /// Thick-restart bound on the `Rpa` arm's live basis, passed straight to
    /// [`SymplecticDavidsonOpts::max_subspace`]: `0` (the default) = the kernel's auto
    /// `max(8·(k+b), 32)` for start-block size `b`, clamped to `dim T^φ`. Too tight a bound
    /// limit-cycles on hard spectra and which arm survives depends on it, so a
    /// molecular channel wants the bound its untyped twin runs at. Ignored by the `Tda` arm.
    pub max_subspace: usize,
}

impl Default for SectorOpts {
    fn default() -> Self {
        SectorOpts {
            tol: 1e-8,
            max_iter: 300,
            readout: SectorReadout::Tda,
            seed: 0x5EC7,
            kernel: RpaKernel::Paired,
            precond: SympPrecond::Auto,
            max_subspace: 0,
        }
    }
}

pub(crate) fn resolve_sector_precond(p: SympPrecond, what: &str) -> SympPrecond {
    let r = p.resolve_sector();
    match r {
        SympPrecond::Off | SympPrecond::DiagUnshifted { .. } => r,
        other => panic!(
            "{what}: SympPrecond::{other:?} has no shifted-channel model on the sector layer — the \
             sector readouts take SympPrecond::Off or SympPrecond::DiagUnshifted (the fixed D0 \
             inverse of the channel's a_diagonal, SectorCtx / Su2SectorCtx), or the default Auto \
             (which resolves to Off)"
        ),
    }
}

const D0_MODEL_SCALE_LO: f64 = 1e-100;
const D0_MODEL_SCALE_HI: f64 = 1e100;

fn guard_d0_model<V: InnerProductVector<Scalar = f64>>(mut m: V) -> Option<V> {
    let mut dmax = 0.0f64;
    m.for_each_coeff_mut(|c| dmax = dmax.max(c.max(0.0)));
    if dmax > 0.0 {
        if !(D0_MODEL_SCALE_LO..=D0_MODEL_SCALE_HI).contains(&dmax) {
            let s = 1.0 / dmax;
            m.for_each_coeff_mut(|c| *c *= s);
        }
        Some(m)
    } else {
        None
    }
}

pub(crate) fn sector_d0_precond<V: InnerProductVector<Scalar = f64>>(
    choice: SympPrecond,
    model: impl FnOnce() -> V,
) -> Option<impl FnMut(&mut V, f64)> {
    let (choice, built) = match choice {
        SympPrecond::DiagUnshifted { .. } => match guard_d0_model(model()) {
            Some(m) => (choice, Some(m)),
            None => (SympPrecond::Off, None),
        },
        other => (other, None),
    };
    build_symp_precond(choice, move || {
        built.expect("sector_d0_precond: the factory reads the model on DiagUnshifted only")
    })
}

pub(crate) fn sector_paired_precond<V: InnerProductVector<Scalar = f64>>(
    choice: PairedPrecond,
    d_a: V,
) -> Option<impl FnMut(&mut V, f64)> {
    match choice {
        PairedPrecond::Unshifted { .. } => {
            guard_d0_model(d_a).and_then(|m| paired_precond(choice, m))
        }
        PairedPrecond::Off => None,
        PairedPrecond::Shifted => unreachable!(
            "sector_paired_precond: the sector layer resolves to Off or the D0 diagonal only"
        ),
    }
}

pub(crate) trait SectorPencil {
    type V: InnerProductVector<Scalar = f64> + Send + Sync;
    fn a_apply(&self, v: &Self::V) -> Self::V;
    fn b_apply(&self, v: &Self::V) -> Self::V;
    fn hpp_apply(&self, v: &Self::V) -> Self::V;
    fn hmm_apply(&self, v: &Self::V) -> Self::V;
    fn a_diagonal(&self) -> Self::V;
    fn a_diagonal_squared(&self) -> Self::V;
    fn clean(v: &mut Self::V);
}

pub(crate) fn check_sector_kernel(kernel: RpaKernel, what: &str) {
    if let RpaKernel::Signed { probe_budget, .. } = kernel {
        assert!(
            probe_budget == 0,
            "{what}: RpaKernel::Signed {{ probe_budget: {probe_budget} }} — the sector layer never \
             runs a PD probe (the lane is the one `metric` names, Auto = Ii); pass \
             RpaKernel::signed(metric)"
        );
    }
}

pub(crate) fn sector_rpa_solve<P: SectorPencil>(
    p: &P,
    v0s: &[P::V],
    nroots: usize,
    space_dim: usize,
    opts: &SectorOpts,
    precond_choice: SympPrecond,
) -> (RpaReadout, Vec<RpaAmplitudes<P::V>>) {
    let dopts = SymplecticDavidsonOpts {
        tol: opts.tol,
        max_basis: opts.max_iter,
        max_subspace: opts.max_subspace,
        precond: precond_choice,
        kernel: opts.kernel,
        ..SymplecticDavidsonOpts::default()
    };
    if let Some((pp, popts)) = paired_route(&dopts) {
        let d_a = p.a_diagonal();
        let mu = paired_mu(PairedHook::Orthogonal, &d_a);
        let precond = sector_paired_precond(pp, d_a);
        let out = omega_davidson_paired(
            |v| paired_image(P::clean, |q| p.a_apply(q), mu, v, true),
            |v| paired_image(P::clean, |q| p.b_apply(q), mu, v, false),
            P::clean,
            precond,
            false,
            v0s,
            &[],
            nroots,
            space_dim,
            &popts,
        );
        return readout_from_paired(out);
    }
    let metric = match opts.kernel {
        RpaKernel::Signed {
            metric: RpaMetric::Rr,
            ..
        } => RpaMetric::Rr,
        _ => RpaMetric::Ii,
    };
    let precond = sector_d0_precond(precond_choice, || p.a_diagonal_squared());
    let applies = std::cell::Cell::new(0usize);
    let bump = || applies.set(applies.get() + 1);
    let deflate = |_v: &mut P::V| {};
    let reproject = P::clean;
    let out = match metric {
        RpaMetric::Rr => omega_davidson_signed(
            |w| {
                bump();
                p.hmm_apply(w)
            },
            |v| {
                bump();
                p.hpp_apply(v)
            },
            deflate,
            reproject,
            precond,
            v0s,
            nroots,
            space_dim,
            &dopts,
            REFUSE_TAU,
        ),
        _ => omega_davidson_signed(
            |w| {
                bump();
                p.hpp_apply(w)
            },
            |v| {
                bump();
                p.hmm_apply(v)
            },
            deflate,
            reproject,
            precond,
            v0s,
            nroots,
            space_dim,
            &dopts,
            REFUSE_TAU,
        ),
    };
    readout_from_signed(out, metric, applies.get())
}

/// The spectral payload of a [`SectorResponse`], tagged by the readout that produced it — because
/// `eig(A^φ)` (an energy) and the symplectic `ω²` are different quantities and a bare `Vec<f64>`
/// carries no unit.
#[derive(Clone, Debug)]
pub enum SectorSpectrum {
    /// `eig(A^φ)` ascending: Ritz values of `H|_{Γ0 ⊕ φ} − λ` compressed to `T^φ`, i.e. channel
    /// excitation ENERGIES relative to `λ` (upper bounds by Cauchy interlacing unless
    /// [`SectorResponse::dim`] equals the target sector's determinant count — module header).
    Tda {
        theta: Vec<f64>,
        negative: Vec<NegativeWitness>,
    },
    /// The RPA solve's [`RpaReadout`]: every imaginary mode the solve found, the `k` lowest
    /// real ones and the complex ones, with the report of the kernel that ran
    /// ([`SectorOpts::kernel`]): the paired kernel's counting offset, or the signed kernel's LANE
    /// and metric events. It comes back with NO mode on TWO routes only, which
    /// `(converged, dim, applies)` separates — see [`SectorCtx::sector_lowest`], "Empty
    /// readouts", for the table. ★ On both empty routes the run returns before any solve and the
    /// report is the REQUESTED kernel's: a zero-offset paired report, or the signed report with
    /// the requested `metric` verbatim (it CAN be [`RpaMetric::Auto`]); wherever `applies > 0` a
    /// signed lane is `Ii` or `Rr`.
    Rpa(RpaReadout),
}

/// The outcome of one channel's readout, with the conditioning of the base point reported beside
/// the numbers (a spectral value at an ill-conditioned frame is not quotable on its own).
#[derive(Clone, Debug)]
pub struct SectorResponse {
    /// The numbers, tagged by readout — read them through [`Self::theta`] / [`Self::omega2`] /
    /// [`Self::omega`] / [`Self::rpa`], which cannot hand back a value in the wrong units.
    pub spectrum: SectorSpectrum,
    /// `dim T^φ` ([`tangent_dim_grids`]) — the exact-exhaustion cap of the channel, STRUCTURAL
    /// (shapes only). Not the target sector's dimension: compare it against the determinant count
    /// of `Γ0 ⊕ φ` before reading the spectrum as that sector's levels (module header).
    pub dim: usize,
    /// The channel's irrep flux.
    pub phi: Irrep,
    /// The irrep the response reaches, `Γ0 ⊕ φ`.
    pub target: Irrep,
    /// Every requested root came back converged (on the `Rpa` arm, the readout's `converged`).
    pub converged: bool,
    /// Outer iterations (`Tda`) / basis vectors built (`Rpa`, either kernel).
    pub iterations: usize,
    /// Applies consumed by the readout kernel. `A^φ` applies on the `Tda` arm (the negative
    /// certificates' fresh applies included); on the `Rpa` arm Hessian-class applies in
    /// `(A, B)` PAIRS — the paired kernel's pairs, or the signed kernel's `H^±` applies counted on
    /// its two operator closures (one `H^±` apply is one pair's work) — the fresh residuals and
    /// certificates the kernel runs included, the preconditioner's diagonal build and any later
    /// mode check or `f_W` excluded.
    pub applies: usize,
    /// Smallest Schmidt value of the BASE point ([`min_schmidt`]) — base-point data, identical
    /// across channels, quoted with every reading.
    pub sigma_min: f64,
}

impl SectorResponse {
    /// The `Tda` arm's `eig(A^φ)`, or `None` on an `Rpa` readout — `ω²` is not an energy.
    pub fn theta(&self) -> Option<&[f64]> {
        match &self.spectrum {
            SectorSpectrum::Tda { theta, .. } => Some(theta),
            SectorSpectrum::Rpa(_) => None,
        }
    }

    /// The `Rpa` arm's readout, or `None` on a `Tda` readout.
    pub fn rpa(&self) -> Option<&RpaReadout> {
        match &self.spectrum {
            SectorSpectrum::Tda { .. } => None,
            SectorSpectrum::Rpa(r) => Some(r),
        }
    }

    /// The `Rpa` arm's signed `ω²` (imaginary and real modes, [`RpaReadout::omega2`]), or `None`
    /// on a `Tda` readout. `Some(vec![])` is a real answer and is NOT the same statement as
    /// `None`: it reports, in `Rpa` units, that this channel has no `ω²` to publish. Two routes
    /// produce it, told apart by `(converged, dim)` — the structurally empty channel (or
    /// `k == 0`) and the annihilated seed block ([`SectorCtx::sector_lowest`], "Empty readouts").
    pub fn omega2(&self) -> Option<Vec<f64>> {
        self.rpa().map(RpaReadout::omega2)
    }

    /// `ω` — the readout's publication gate ([`RpaReadout::omega`]): `Some` only on a converged,
    /// non-empty `Rpa` readout with no anomaly (no imaginary or complex mode, no de-excitation, no
    /// metric event), `None` otherwise. At `B^φ ≡ 0` a de-excitation to a LOWER state of
    /// `Γ0 ⊕ φ` is a positive `θ = a²` with a negative Krein margin, with NO negative `ω²`
    /// anywhere — the bare positivity rule would publish it as the real `|a|`, i.e. a state at
    /// `λ + |a|` for one at `λ − |a|`. Publish an anomalous solve mode by mode through
    /// [`SectorCtx::rpa_mode_checks`].
    pub fn omega(&self) -> Option<Vec<f64>> {
        self.rpa()?.omega()
    }

    /// The negativity witnesses of either arm: the `Tda` arm's negative energies, or the `Rpa`
    /// readout's imaginary-mode certificates ([`RpaMode::certificate`]), in mode order.
    pub fn negatives(&self) -> Vec<NegativeWitness> {
        match &self.spectrum {
            SectorSpectrum::Tda { negative, .. } => negative.clone(),
            SectorSpectrum::Rpa(r) => r.modes.iter().filter_map(|m| m.certificate).collect(),
        }
    }

    /// Every negative value carries a certified fresh-apply witness (vacuously true when there
    /// are none) — the publication gate, on either arm.
    pub fn all_negatives_certified(&self) -> bool {
        match &self.spectrum {
            SectorSpectrum::Tda { negative, .. } => negative.iter().all(NegativeWitness::certified),
            SectorSpectrum::Rpa(r) => r.all_negatives_certified(),
        }
    }

    /// The SIGNED solve met an indefinite metric (a metric event in its report): its `converged`
    /// does not license its `ω` — read [`SectorCtx::rpa_mode_checks`]. `false` on a `Tda` readout
    /// and ALWAYS `false` on a paired readout, which has no metric: read [`Self::anomalous`].
    pub fn metric_indefinite(&self) -> bool {
        self.rpa().is_some_and(|r| r.metric_events() > 0)
    }

    /// The `Rpa` solve found anything a stable channel would not show ([`RpaReadout::anomalous`]:
    /// an imaginary or complex mode, a de-excitation, a negativity certificate, a metric event or
    /// a nonzero counting offset) — the kernel-independent flag; `ω` is then withheld
    /// ([`Self::omega`]) and published mode by mode through [`SectorCtx::rpa_mode_checks`]. `false`
    /// on a `Tda` readout.
    pub fn anomalous(&self) -> bool {
        self.rpa().is_some_and(RpaReadout::anomalous)
    }

    /// `#imaginary + 2·#complex + 2·#{real, Krein margin < 0}` — the inertia of the PROJECTED
    /// pencil the solve diagonalized, and so a LOWER BOUND on `n₋(A^φ + B^φ) + n₋(A^φ − B^φ)` at
    /// ANY iteration (the explored span can only miss negative directions). Equality holds at
    /// exhaustion ONLY when every Ritz value of the exhausted span is LISTED — `k ≥ dim − #neg`
    /// (the list keeps the imaginary modes plus the `k` lowest real ones, so a smaller `k` drops
    /// the higher de-excitations and complex pairs from the count). Each count in its own unit: an
    /// imaginary mode once, each complex mode (one member per conjugate pair, and only the WANTED
    /// ones) twice, each de-excitation twice; a zero `ω²` or a Krein-neutral real mode counts
    /// nothing. `None` on a `Tda` readout (`eig(A^φ)` has no pencil); `Some(0)` on an empty or
    /// anomaly-free `Rpa` readout.
    pub fn index_lower_bound(&self) -> Option<usize> {
        let r = self.rpa()?;
        let count = |f: &dyn Fn(&RpaMode) -> bool| r.modes.iter().filter(|m| f(m)).count();
        let imaginary = count(&|m| m.kind == RpaModeKind::Imaginary);
        let complex = count(&|m| m.kind == RpaModeKind::Complex);
        let de_excitations =
            count(&|m| m.kind == RpaModeKind::Real && m.omega.re > 0.0 && m.krein < 0.0);
        Some(imaginary + 2 * complex + 2 * de_excitations)
    }
}

struct ShiftedEnvs<T: crate::scalar::Scalar> {
    u: Arc<TtVector<T>>,
    v: Arc<TtVector<T>>,
    l: Vec<Sketch<T>>,
    r_phi: Vec<Sketch<T>>,
    ws_u: Vec<OneSiteWorkspace>,
    ws_v: Vec<OneSiteWorkspace>,
    ws_du: Vec<OneSiteWorkspace>,
}

/// One response channel bound at a fixed reference: the frames and `λ` of a [`WeingartenCtx`], the
/// flux `φ`, and the Hamiltonian rebuilt on `space_out = space.with_target(Γ0 ⊕ φ)`.
pub struct SectorCtx<'c, 'h, T: crate::scalar::Scalar> {
    ctx: &'c WeingartenCtx<'h, T>,
    h_out: &'h SparseHamiltonian<T>,
    phi: Irrep,
    envs: std::sync::OnceLock<ShiftedEnvs<T>>,
}

impl SectorPencil for SectorCtx<'_, '_, f64> {
    type V = TtTangent<f64>;
    fn a_apply(&self, v: &TtTangent<f64>) -> TtTangent<f64> {
        SectorCtx::a_apply(self, v)
    }
    fn b_apply(&self, v: &TtTangent<f64>) -> TtTangent<f64> {
        SectorCtx::b_apply(self, v)
    }
    fn hpp_apply(&self, v: &TtTangent<f64>) -> TtTangent<f64> {
        SectorCtx::hpp_apply(self, v)
    }
    fn hmm_apply(&self, v: &TtTangent<f64>) -> TtTangent<f64> {
        SectorCtx::hmm_apply(self, v)
    }
    fn a_diagonal(&self) -> TtTangent<f64> {
        SectorCtx::a_diagonal(self)
    }
    fn a_diagonal_squared(&self) -> TtTangent<f64> {
        SectorCtx::a_diagonal_squared(self)
    }
    fn clean(v: &mut TtTangent<f64>) {
        v.gauge_fix_orthogonal_mut();
    }
}

impl<'c, 'h> SectorCtx<'c, 'h, f64> {
    /// Bind the `φ` channel at the reference `ctx` was built on.
    pub fn new(
        ctx: &'c WeingartenCtx<'h, f64>,
        h_out: &'h SparseHamiltonian<f64>,
        phi: Irrep,
    ) -> Self {
        assert_ne!(
            phi,
            Irrep::IDENTITY,
            "SectorCtx: phi = IDENTITY is not a cross-sector channel — T^IDENTITY is x's ordinary              tangent space and A^IDENTITY its ordinary effective Hamiltonian. Use              weingarten_operator(h, x) and the shipped in-sector RPA/TDA readouts"
        );
        let (base_l, _base_r) = ctx.bases();
        let space = base_l.space;
        let space_out = space.with_target(space.target.fuse(phi));
        super::assert_shifted_operator(h_out, base_l, phi);
        debug_assert_eq!(h_out.space, space_out);
        SectorCtx {
            ctx,
            h_out,
            phi,
            envs: std::sync::OnceLock::new(),
        }
    }

    /// The channel's irrep flux.
    #[inline]
    pub fn phi(&self) -> Irrep {
        self.phi
    }

    /// The Rayleigh quotient at the reference — the shift in `A^φ = P^φ(H_out − λ)P^φ`.
    #[inline]
    pub fn lambda(&self) -> f64 {
        self.ctx.lambda
    }

    /// `space_out = x.space.with_target(Γ0 ⊕ φ)`: the space `ι^φ(ξ)` and `h_out` live in.
    #[inline]
    pub fn space_out(&self) -> QnSpace {
        self.h_out.space
    }

    /// The ORIGINAL gauge frames of `x`, in `x`'s own space — not the shifted pair, which
    /// the crate-internal `shifted_frames` builds internally wherever it is needed.
    #[inline]
    pub fn bases(&self) -> (&TtVector<f64>, &TtVector<f64>) {
        self.ctx.bases()
    }

    #[inline]
    pub(super) fn weingarten(&self) -> &'c WeingartenCtx<'h, f64> {
        self.ctx
    }

    fn envs(&self) -> &ShiftedEnvs<f64> {
        self.envs.get_or_init(|| {
            let h = self.h_out;
            let (bl, br) = self.bases();
            let (u, v) = shifted_frames(bl, br, self.phi);
            let space_out = u.space;
            let d = space_out.d;
            let r = &u.ranks;
            let r_col = &v.ranks;

            let ws_u: Vec<OneSiteWorkspace> =
                (0..d).map(|k| onesite_workspace(h, u.core(k))).collect();
            let ws_v: Vec<OneSiteWorkspace> =
                (0..d).map(|k| onesite_workspace(h, v.core(k))).collect();
            let ws_du: Vec<OneSiteWorkspace> = (0..d)
                .map(|k| {
                    let decl = SparseCore::<f64>::zeros(space_out, k, &r[k], &r_col[k + 1]);
                    onesite_workspace(h, &decl)
                })
                .collect();

            let l = left_to_right_sketching(h, &u);
            let r_phi = right_to_left_sketching(h, &v);

            ShiftedEnvs {
                u: Arc::new(u),
                v: Arc::new(v),
                l,
                r_phi,
                ws_u,
                ws_v,
                ws_du,
            }
        })
    }

    /// `P^φ(y)` for `y ∈ space_out` — [`TtTangent::project_shifted`] at these frames, run on the
    /// channel bundle's already-dressed pair.
    pub fn project(&self, y: &TtVector<f64>) -> TtTangent<f64> {
        let e = self.envs();
        TtTangent::project_shifted_at_frames(&e.u, &e.v, self.phi, y)
    }

    /// `A^φ·ξ = P^φ((H_out − λ)·ι^φ(ξ))` — the FUSED environment sweep of
    /// [`Self::a_apply_fused`], with `QNTT_A_APPLY_EXACT` restoring the unfused reference product
    /// [`Self::a_apply_exact`].
    pub fn a_apply(&self, xi: &TtTangent<f64>) -> TtTangent<f64> {
        if crate::knobs::A_APPLY_EXACT.get() {
            return self.a_apply_exact(xi);
        }
        self.a_apply_fused(xi)
    }

    /// `A^φ·ξ` as a FUSED environment sweep — **`H_out·ι^φ(ξ)` is never formed** — bypassing the
    /// `QNTT_A_APPLY_EXACT` switch. The entry point the equivalence tests pin against
    /// [`Self::a_apply_exact`] / [`effham_apply_shifted`].
    pub fn a_apply_fused(&self, xi: &TtTangent<f64>) -> TtTangent<f64> {
        assert_eq!(
            xi.phi(),
            self.phi,
            "SectorCtx::a_apply_fused: the direction carries flux {:?} but the channel is {:?}",
            xi.phi(),
            self.phi
        );
        let h = self.h_out;
        let e = self.envs();
        let d = e.u.space.d;
        let r = e.u.ranks.clone();
        let r_col = e.v.ranks.clone();
        debug_assert_eq!(
            xi.rank(),
            &r[..],
            "a_apply_fused: the direction's row grids are not the channel's"
        );
        debug_assert_eq!(
            xi.rank_col(),
            &r_col[..],
            "a_apply_fused: the direction's column grids are not the channel's"
        );

        if shifted_channel_is_empty(&e.r_phi[d - 1]) {
            let mut out = TtTangent::zero_shifted(self.phi, r, r_col, e.u.clone(), e.v.clone());
            out.axpy(-self.lambda(), xi);
            return out;
        }

        let components = FusedTermA {
            h,
            u: &e.u,
            v: &e.v,
            l: &e.l,
            r: &e.r_phi,
            ws_u: &e.ws_u,
            ws_v: &e.ws_v,
            ws_du: &e.ws_du,
        }
        .components(&xi.components);

        let mut out =
            TtTangent::from_parts_shifted(r, r_col, self.phi, e.u.clone(), e.v.clone(), components);
        out.axpy(-self.lambda(), xi);
        out
    }

    /// `A^φ·ξ` by the UNFUSED reference product — [`effham_apply_shifted`] at these frames, `φ`
    /// and `λ`, i.e. embed at doubled rank, apply `h_out` exactly, project, subtract `λ·ξ`.
    pub fn a_apply_exact(&self, xi: &TtTangent<f64>) -> TtTangent<f64> {
        let (bl, br) = self.bases();
        effham_apply_shifted(self.h_out, bl, br, self.phi, self.lambda(), xi)
    }

    /// The channel's SHIFTED effective-Hamiltonian diagonal `D_A^φ`, as a tangent of `T^φ`'s shape
    /// — the shifted twin of [`WeingartenCtx::a_diagonal`] and the model behind the sector
    /// readouts' `D0^φ` preconditioner ([`SectorOpts::precond`]).
    pub fn a_diagonal(&self) -> TtTangent<f64> {
        let h = self.h_out;
        let e = self.envs();
        let space_out = e.u.space;
        let d = space_out.d;
        let r = e.u.ranks.clone();
        let r_col = e.v.ranks.clone();
        let shift = -self.lambda();
        if shifted_channel_is_empty(&e.r_phi[d - 1]) {
            let mut diag = TtTangent::zero_shifted(self.phi, r, r_col, e.u.clone(), e.v.clone());
            diag.for_each_coeff_mut(|c| *c += shift);
            return diag;
        }
        let components: Vec<SparseCore<f64>> = (0..d)
            .map(|k| {
                let decl = SparseCore::<f64>::zeros(space_out, k, &r[k], &r_col[k + 1]);
                h.effective_diagonal(&e.l[k], &e.r_phi[k], &decl, &e.ws_du[k])
            })
            .collect();
        let mut diag =
            TtTangent::from_parts_shifted(r, r_col, self.phi, e.u.clone(), e.v.clone(), components);
        diag.for_each_coeff_mut(|c| *c += shift);
        diag
    }

    pub(super) fn a_diagonal_squared(&self) -> TtTangent<f64> {
        let mut d = self.a_diagonal();
        d.for_each_coeff_mut(|c| {
            let sq = *c * *c;
            *c = sq;
        });
        d
    }

    /// The row-major `n×n` Gram of `A^φ` over `basis` — `n` FUSED applies plus `n²` tangent dots,
    /// the A-half of [`Self::grams`] exactly.
    pub fn a_gram(&self, basis: &[TtTangent<f64>]) -> Vec<f64> {
        for (i, b) in basis.iter().enumerate() {
            assert_eq!(
                b.phi(),
                self.phi,
                "SectorCtx::a_gram: basis vector {i} carries flux {:?} but the channel is {:?} — \
                 a Gram mixing fluxes is not the channel's operator",
                b.phi(),
                self.phi
            );
        }
        let n = basis.len();
        let pure: Vec<TtTangent<f64>> =
            basis.iter().map(|u| self.project(&u.to_vector())).collect();
        let mut a = vec![0.0f64; n * n];
        for (j, uj) in pure.iter().enumerate() {
            let aj = self.a_apply(uj);
            for (i, ui) in pure.iter().enumerate() {
                a[i * n + j] = ui.dot(&aj);
            }
        }
        a
    }

    /// The row-major `n×n` Gram of `A^φ` by the UNFUSED reference route —
    /// [`effham_gram_shifted`] at these frames. The oracle [`Self::a_gram`] is pinned against.
    pub fn a_gram_exact(&self, basis: &[TtTangent<f64>]) -> Vec<f64> {
        let (bl, br) = self.bases();
        effham_gram_shifted(self.h_out, bl, br, self.phi, self.lambda(), basis)
    }

    /// An orthonormal basis of `T^φ` — [`tangent_basis_shifted`]. `O(dim)` tangents live at once;
    /// the iterative readout below does not use it.
    pub fn basis(&self) -> Vec<TtTangent<f64>> {
        let (bl, br) = self.bases();
        tangent_basis_shifted(bl, br, self.phi)
    }

    /// `dim T^φ` — [`tangent_dim_grids`], structural (shapes only). The MGS cull can leave
    /// [`Self::basis`] SHORTER at a truncated or specially-structured point.
    pub fn dim(&self) -> usize {
        let (bl, _) = self.bases();
        tangent_dim_grids(&bl.space, &bl.ranks, self.phi)
    }

    /// The `k` lowest eigenvalues of `A^φ` — the channel's TDA-like excitation energies — by block
    /// Davidson restricted to `range(P^φ)`.
    pub fn sector_lowest_tda(&self, k: usize, opts: &SectorOpts) -> SectorResponse {
        assert!(
            opts.readout == SectorReadout::Tda,
            "sector_lowest_tda: the symplectic (RPA) readout of the pair (A^phi, B^phi) — Phase B \
             of the sector-response campaign — is \
             sector_lowest's; this entry point reports eig(A^phi) only, and returning an energy \
             under an RPA label would mislabel the quantity"
        );
        let precond_choice = resolve_sector_precond(opts.precond, "sector_lowest_tda");
        let (base_l, base_r) = self.bases();
        let sigma_min = min_schmidt(base_l, base_r);
        let dim = self.dim();
        let empty = |converged: bool| SectorResponse {
            spectrum: SectorSpectrum::Tda {
                theta: Vec::new(),
                negative: Vec::new(),
            },
            dim,
            phi: self.phi,
            target: self.space_out().target,
            converged,
            iterations: 0,
            applies: 0,
            sigma_min,
        };
        let nroots = k.min(dim);
        if nroots == 0 {
            return empty(true);
        }

        let v0s = self.seed_block(nroots + 2, opts.seed);
        if v0s.is_empty() {
            return empty(false);
        }

        let dmodel = self.a_diagonal();
        let mu = super::complement_shift(super::diag_inf_scale(&dmodel));
        let mut d0 = sector_d0_precond(precond_choice, move || dmodel);
        let mut applies = 0usize;
        let mut op = |xi: &TtTangent<f64>| {
            applies += 1;
            let mut q = xi.clone();
            q.gauge_fix_orthogonal_mut();
            let mut aq = self.a_apply(&q);
            super::add_complement_shift(&mut aq, xi, q, mu);
            aq
        };
        let mut precond = |t: &mut TtTangent<f64>, theta: f64| {
            if let Some(p) = d0.as_mut() {
                p(t, theta);
            }
        };
        let mut project = |t: &mut TtTangent<f64>| t.gauge_fix_orthogonal_mut();
        let dopts = BlockDavidsonOpts {
            tol: opts.tol,
            maxiter: opts.max_iter,
            ..BlockDavidsonOpts::for_roots(nroots)
        };
        let (theta, us, info) =
            davidson_block_projected(&mut op, &v0s, &mut precond, &mut project, nroots, &dopts);

        let (negative, _certify_ops) = certify_negatives(
            &theta,
            theta.len(),
            |j| us.get(j).cloned(),
            &mut |v: &TtTangent<f64>| {
                applies += 1;
                self.a_apply(v)
            },
            &mut |v: &TtTangent<f64>| v.clone(),
            &mut |_v: &mut TtTangent<f64>| {},
            &mut |v: &mut TtTangent<f64>| v.gauge_fix_orthogonal_mut(),
        );

        SectorResponse {
            spectrum: SectorSpectrum::Tda { theta, negative },
            dim,
            phi: self.phi,
            target: self.space_out().target,
            converged: info.converged,
            iterations: info.niter,
            applies,
            sigma_min,
        }
    }

    pub(super) fn seed_block(&self, n: usize, seed: u64) -> Vec<TtTangent<f64>> {
        let (base_l, _) = self.bases();
        let space_out = self.space_out();
        let d = space_out.d;
        let r = &base_l.ranks;
        let grid: Vec<RankGrid> = (0..=d)
            .map(|b| {
                let shifted = r[b].relabel_irrep(self.phi);
                let mut g = RankGrid::zeros(&space_out);
                for s in space_out.bond_sectors(b) {
                    g[s] = match b {
                        0 => r[b][s],
                        _ if b == d => shifted[s],
                        _ => r[b][s] + shifted[s],
                    };
                }
                g
            })
            .collect();
        let mut rng = Xoshiro256PlusPlus::seed_from_u64(seed);
        (0..n)
            .filter_map(|_| {
                let g = tt_randn::<f64, _>(space_out, &grid, &mut rng);
                let n0 = g.norm();
                let v = self.project(&g);
                (v.norm() > f64::EPSILON.sqrt() * n0).then_some(v)
            })
            .collect()
    }

    /// `B^φ·ξ` — the flux-`φ` block of the Weingarten (shape-operator) map
    /// `B''ξ = P''(D_ξ P'')r⊥` of the point-group-UNTYPED manifold `M''` through `x`.
    pub fn b_apply(&self, xi: &TtTangent<f64>) -> TtTangent<f64> {
        assert_eq!(
            xi.phi(),
            self.phi,
            "SectorCtx::b_apply: the direction carries flux {:?} but the channel is {:?}",
            xi.phi(),
            self.phi
        );
        self.ctx.b_apply_any(xi)
    }

    /// The whole Riemannian Hessian block `H^rr = A^φ·ξ + B^φ·ξ`.
    pub fn hess_apply(&self, xi: &TtTangent<f64>) -> TtTangent<f64> {
        self.a_apply(xi).add(&self.b_apply(xi))
    }

    /// `H^rr = A^φ + B^φ` — an alias of [`Self::hess_apply`], named for the symplectic driver's
    /// outer factor.
    pub fn hpp_apply(&self, xi: &TtTangent<f64>) -> TtTangent<f64> {
        self.hess_apply(xi)
    }

    /// `H^ii = A^φ − B^φ` — the symplectic driver's inner factor.
    pub fn hmm_apply(&self, xi: &TtTangent<f64>) -> TtTangent<f64> {
        self.a_apply(xi).sub(&self.b_apply(xi))
    }

    /// The row-major `n×n` Grams `A_ij = ⟨b_i, A^φ b_j⟩` and `B_ij = ⟨b_i, B^φ b_j⟩` over a
    /// `⟨·,·⟩`-orthonormal basis of `T^φ` — the shifted twin of
    /// [`crate::solvers::hess::hessian_grams`].
    pub fn grams(&self, basis: &[TtTangent<f64>]) -> (Vec<f64>, Vec<f64>) {
        for (i, b) in basis.iter().enumerate() {
            assert_eq!(
                b.phi(),
                self.phi,
                "SectorCtx::grams: basis vector {i} carries flux {:?} but the channel is {:?} — \
                 a Gram mixing fluxes is not the channel's operator",
                b.phi(),
                self.phi
            );
        }
        let n = basis.len();
        let pure: Vec<TtTangent<f64>> =
            basis.iter().map(|u| self.project(&u.to_vector())).collect();
        let mut a = vec![0.0f64; n * n];
        let mut b = vec![0.0f64; n * n];
        for (j, uj) in pure.iter().enumerate() {
            let aj = self.a_apply(uj);
            let bj = self.b_apply(uj);
            for (i, ui) in pure.iter().enumerate() {
                a[i * n + j] = ui.dot(&aj);
                b[i * n + j] = ui.dot(&bj);
            }
        }
        (a, b)
    }

    /// The DENSE symplectic (RPA) spectrum of the pair `(A^φ, B^φ)` over `basis` — the certified
    /// oracle the iterative [`Self::sector_lowest`] must reproduce.
    pub fn symplectic_eigenvalues(&self, basis: &[TtTangent<f64>]) -> SymplecticSpectrum {
        let (a, b) = self.grams(basis);
        symplectic_from_grams(&a, &b, basis.len(), 1e-10)
    }

    /// The `k` lowest symplectic (RPA) values of the channel — matrix-free, by the kernel
    /// [`SectorOpts::kernel`] names: the PAIRED-basis Davidson by default, the signed
    /// (hyperbolic) W-metric block Davidson through `RpaKernel::signed(lane)` — reported as SIGNED
    /// `ω²` with the modes' Krein types: [`Self::sector_lowest_vectors`] without the amplitudes.
    pub fn sector_lowest(&self, k: usize, opts: &SectorOpts) -> SectorResponse {
        self.sector_lowest_vectors(k, opts).0
    }

    /// [`Self::sector_lowest`] that also returns each mode's **amplitudes** `(X, Y)`
    /// ([`RpaAmplitudes`]), one per entry in the same order, already inside `T^φ`: the pair the
    /// paired kernel solved for, or the signed kernel's metric-side Ritz vector `q` completed by its
    /// own W-image (`p = Wq/ω`). Zero extra applies. Their consumers are the per-mode acceptance
    /// numbers ([`Self::rpa_mode_checks`]) and the Weingarten fraction `f_W = |⟨q, B^φ q⟩| / |⟨q, A^φ q⟩|` on `q`
    /// ([`RpaAmplitudes::metric_side`] on the readout's [`RpaReadout::side`]; two applies per mode
    /// — what the C ABI's `qntt_f64_sector_rpa` reports beside every `ω²`). A signed solve's
    /// signature −1 entry has `⟨q, W q⟩ < 0`: it is itself a witness that the metric block is
    /// indefinite. Both empty routes return no mode.
    pub fn sector_lowest_vectors(
        &self,
        k: usize,
        opts: &SectorOpts,
    ) -> (SectorResponse, Vec<RpaAmplitudes<TtTangent<f64>>>) {
        assert!(
            opts.readout == SectorReadout::Rpa,
            "sector_lowest: the TDA readout eig(A^phi) — an ENERGY — is sector_lowest_tda's; \
             this entry point reports the symplectic omega^2 of the pair (A^phi, B^phi), and \
             returning one under the other's label would mislabel the quantity"
        );
        let precond_choice = resolve_sector_precond(opts.precond, "sector_lowest");
        check_sector_kernel(opts.kernel, "sector_lowest");
        let (base_l, base_r) = self.bases();
        let sigma_min = min_schmidt(base_l, base_r);
        let dim = self.dim();
        let empty = |converged: bool| SectorResponse {
            spectrum: SectorSpectrum::Rpa(RpaReadout::empty(converged, opts.kernel)),
            dim,
            phi: self.phi,
            target: self.space_out().target,
            converged,
            iterations: 0,
            applies: 0,
            sigma_min,
        };
        let nroots = k.min(dim);
        if nroots == 0 {
            return (empty(true), Vec::new());
        }
        let v0s = self.seed_block(nroots + 2, opts.seed);
        if v0s.is_empty() {
            return (empty(false), Vec::new());
        }

        let (readout, modes) = sector_rpa_solve(self, &v0s, nroots, dim, opts, precond_choice);
        let resp = SectorResponse {
            dim,
            phi: self.phi,
            target: self.space_out().target,
            converged: readout.converged,
            iterations: readout.iterations,
            applies: readout.applies,
            sigma_min,
            spectrum: SectorSpectrum::Rpa(readout),
        };
        (resp, modes)
    }

    /// The per-mode acceptance numbers of a sector RPA solve (either kernel) — one [`ModeCheck`]
    /// per `omega2` entry of `resp`, from the amplitudes [`Self::sector_lowest_vectors`] returned
    /// with it (same order, same channel). Every number is defined in [`crate::solvers::hess::rpa_mode_checks`]'s
    /// module docs (`η`, `residual_fresh`, `κ`, `δω_est`, the excited state's `e_state`, `ΔE`
    /// beside `ΔE_pair`, `σ_T`, the opt-in `σ`, `f_W`, the gauge defect); this is the SHIFTED
    /// pencil's spelling of that one body (`mode_checks_core`):
    ///
    /// # Panics
    ///
    /// On a `Tda` response (`eig(A^φ)` has no pencil, no metric-side vector and no Krein signature)
    /// and on `modes.len() != omega2.len()`.
    pub fn rpa_mode_checks(
        &self,
        resp: &SectorResponse,
        modes: &[RpaAmplitudes<TtTangent<f64>>],
        opts: &ModeCheckOpts,
    ) -> Vec<ModeCheck> {
        let _ = opts;
        let readout = match &resp.spectrum {
            SectorSpectrum::Rpa(r) => r,
            SectorSpectrum::Tda { .. } => panic!(
                "SectorCtx::rpa_mode_checks: a Tda response — eig(A^phi), an ENERGY — has no \
                 pencil, no metric-side Ritz vector and no Krein signature to check; pass the Rpa \
                 readout of sector_lowest_vectors and its vectors"
            ),
        };
        assert_eq!(
            modes.len(),
            readout.modes.len(),
            "rpa_mode_checks: one amplitude pair per mode"
        );
        if readout.modes.is_empty() {
            return Vec::new();
        }
        let lane = readout.side();
        let hpp = |v: &TtTangent<f64>| self.hpp_apply(v);
        let hmm = |v: &TtTangent<f64>| self.hmm_apply(v);
        let (w, r, y_sign): (&TanOp<'_, TtTangent<f64>>, &TanOp<'_, TtTangent<f64>>, f64) =
            match lane {
                RpaMetric::Rr => (&hpp, &hmm, -1.0),
                RpaMetric::Ii => (&hmm, &hpp, 1.0),
                RpaMetric::Auto => panic!(
                    "SectorCtx::rpa_mode_checks: a non-empty Rpa readout whose lane was never \
                     resolved"
                ),
            };
        let a = |v: &TtTangent<f64>| self.a_apply(v);
        let b = |v: &TtTangent<f64>| self.b_apply(v);
        let reproject = |v: &mut TtTangent<f64>| *v = self.project(&v.to_vector());
        let deflate = |_v: &mut TtTangent<f64>| {};
        let ops = ModeCheckOps {
            w,
            r,
            a: &a,
            b: &b,
            reproject: &reproject,
            deflate: &deflate,
            y_sign,
        };
        mode_checks_core(&readout.modes, modes, &ops)
    }
}

#[cfg(test)]
mod t8s_rpa_readout_tests {
    use super::{SectorResponse, SectorSpectrum};
    use crate::pointgroup::Irrep;
    use crate::scalar::C64;
    use crate::solvers::hess::{
        MetricEvent, NegativeWitness, RpaMetric, RpaMode, RpaModeKind, RpaReadout, RpaReport,
    };

    fn rpa(omega2: Vec<f64>, negative: Vec<NegativeWitness>) -> SectorResponse {
        let n = omega2.len();
        signed(omega2, vec![1; n], negative, None)
    }

    struct Report {
        n_neg_basis: usize,
        general_path: bool,
        complex: Vec<(f64, f64)>,
    }

    fn signed(
        omega2: Vec<f64>,
        signature: Vec<i8>,
        negative: Vec<NegativeWitness>,
        report: Option<Report>,
    ) -> SectorResponse {
        let mut modes: Vec<RpaMode> = omega2
            .iter()
            .zip(&signature)
            .map(|(&w2, &s)| {
                let imaginary = w2 < 0.0;
                RpaMode {
                    omega: C64::new(w2, 0.0).sqrt(),
                    omega2: C64::new(w2, 0.0),
                    kind: if imaginary {
                        RpaModeKind::Imaginary
                    } else {
                        RpaModeKind::Real
                    },
                    krein: if imaginary { 0.0 } else { f64::from(s) },
                    residual: 1e-10,
                    certificate: None,
                }
            })
            .collect();
        for w in negative {
            modes[w.index].certificate = Some(w);
        }
        let (events, n_negative_basis, general_path) = match &report {
            None => (Vec::new(), 0, false),
            Some(r) => (events(r.n_neg_basis), r.n_neg_basis, r.general_path),
        };
        for &(re, im) in report.iter().flat_map(|r| &r.complex) {
            let z = C64::new(re, im);
            modes.push(RpaMode {
                omega: z.sqrt(),
                omega2: z,
                kind: RpaModeKind::Complex,
                krein: 0.0,
                residual: 1e-9,
                certificate: None,
            });
        }
        let readout = RpaReadout {
            passed_over: Vec::new(),
            modes,
            converged: true,
            iterations: 4,
            applies: 9,
            report: RpaReport::Signed {
                metric: RpaMetric::Rr,
                events,
                n_negative_basis,
                general_path,
                diag: None,
            },
        };
        SectorResponse {
            spectrum: SectorSpectrum::Rpa(readout),
            dim: 2,
            phi: Irrep::new(1),
            target: Irrep::new(1),
            converged: true,
            iterations: 4,
            applies: 9,
            sigma_min: 0.5,
        }
    }

    fn events(n_neg_basis: usize) -> Vec<MetricEvent> {
        let ev = |accepted: bool| MetricEvent {
            basis_len: 3,
            gn_before: 1.0,
            gn_after: 0.9,
            wnorm2: if accepted { -0.2 } else { -4e-7 },
            accepted,
        };
        if n_neg_basis == 0 {
            vec![ev(false)]
        } else {
            (0..n_neg_basis).map(|_| ev(true)).collect()
        }
    }

    fn report(n_neg_basis: usize, general_path: bool, complex: Vec<(f64, f64)>) -> Report {
        Report {
            n_neg_basis,
            general_path,
            complex,
        }
    }

    fn pair(re: f64, im: f64) -> (f64, f64) {
        (re, im)
    }

    #[test]
    fn t8s_omega_refuses_a_flagged_solve() {
        let flag_only = signed(
            vec![2.0, 12.0],
            vec![1, 1],
            Vec::new(),
            Some(report(0, false, Vec::new())),
        );
        assert!(flag_only.metric_indefinite());
        assert_eq!(
            flag_only.omega2(),
            Some(vec![2.0, 12.0]),
            "the signed omega^2 of a flagged solve IS published"
        );
        assert!(
            flag_only.omega().is_none(),
            "a flagged solve's converged does not license its omega (Q5) — read rpa_mode_checks"
        );
        let held = signed(
            vec![2.0, 12.0],
            vec![1, -1],
            Vec::new(),
            Some(report(1, false, Vec::new())),
        );
        assert!(held.metric_indefinite());
        assert!(
            held.omega().is_none(),
            "an s = -1 entry is a de-excitation, never a published real omega"
        );
        let sig_only = signed(vec![2.0, 12.0], vec![1, -1], Vec::new(), None);
        assert!(!sig_only.metric_indefinite());
        assert!(
            sig_only.omega().is_none(),
            "the s = -1 clause must hold on its own (a report-blind rule would publish |a| here)"
        );
        let clean = signed(vec![2.0, 12.0], vec![1, 1], Vec::new(), None);
        assert!(!clean.metric_indefinite());
        let w = clean
            .omega()
            .expect("unflagged, all positive, all s = +1, no witness");
        assert!(
            (w[0] - 2.0f64.sqrt()).abs() <= 1e-15 && (w[1] - 12.0f64.sqrt()).abs() <= 1e-15,
            "{w:?}"
        );
    }

    #[test]
    fn t8s_index_lower_bound_counts_the_inertia() {
        let five = signed(
            vec![-1.0, 2.0, 3.0],
            vec![-1, -1, 1],
            Vec::new(),
            Some(report(2, true, vec![pair(1.5, 0.3)])),
        );
        assert_eq!(five.index_lower_bound(), Some(5));
        let eight = signed(
            vec![-2.0, -1.0, 0.5, 3.0],
            vec![-1, 1, -1, 1],
            Vec::new(),
            Some(report(2, true, vec![pair(1.5, 0.3), pair(2.5, 0.1)])),
        );
        assert_eq!(eight.index_lower_bound(), Some(8));
        let strict = signed(
            vec![0.0, 1.0],
            vec![-1, 1],
            Vec::new(),
            Some(report(1, false, Vec::new())),
        );
        assert_eq!(
            strict.index_lower_bound(),
            Some(0),
            "omega^2 > 0 is strict: a zero entry carries no inertia"
        );
        assert_eq!(rpa(vec![0.4, 0.9], Vec::new()).index_lower_bound(), Some(0));
        assert_eq!(rpa(Vec::new(), Vec::new()).index_lower_bound(), Some(0));
        let tda = SectorResponse {
            spectrum: SectorSpectrum::Tda {
                theta: vec![-0.5, 0.2],
                negative: Vec::new(),
            },
            ..rpa(Vec::new(), Vec::new())
        };
        assert_eq!(
            tda.index_lower_bound(),
            None,
            "eig(A^phi) has no pencil inertia"
        );
    }

    fn certified() -> NegativeWitness {
        NegativeWitness {
            index: 0,
            theta: -1.0,
            fresh: -1.0,
            fresh_w: 1.0,
        }
    }

    #[test]
    fn t8s_a_certified_resonance_publishes_omega2_and_a_witness_but_never_omega() {
        let r = rpa(vec![-1.0, 3.0], vec![certified()]);
        assert_eq!(
            r.omega2(),
            Some(vec![-1.0, 3.0]),
            "the signed ω² IS published — this route is not an empty readout"
        );
        assert!(
            r.all_negatives_certified(),
            "fresh < 0 ⇒ the resonance claim is variationally certified"
        );
        assert!(
            r.omega().is_none(),
            "ω = i√|ω²| cannot be published as a real number, certified or not"
        );
        assert!(r.theta().is_none(), "ω² is not an energy");
        assert!(r.converged, "a resonance is physics, not a solver failure");
    }

    #[test]
    fn t8s_an_uncertified_negative_is_told_apart_only_by_the_certificate() {
        let bad = rpa(
            vec![-1.0, 3.0],
            vec![NegativeWitness {
                index: 0,
                theta: -1.0,
                fresh: 0.25,
                fresh_w: 1.0,
            }],
        );
        assert_eq!(bad.omega2(), Some(vec![-1.0, 3.0]));
        assert!(!bad.all_negatives_certified());
        assert!(bad.omega().is_none());
        let nan = rpa(
            vec![-1.0, 3.0],
            vec![NegativeWitness {
                index: 0,
                theta: -1.0,
                fresh: f64::NAN,
                fresh_w: f64::NAN,
            }],
        );
        assert!(!nan.all_negatives_certified(), "NaN < 0 is false");
    }

    #[test]
    fn t8s_omega_publishes_only_on_positive_non_empty_and_witness_free() {
        let good = rpa(vec![2.0, 12.0], Vec::new());
        let w = good.omega().expect("all clauses hold");
        assert!(
            (w[0] - 2.0f64.sqrt()).abs() <= 1e-15 && (w[1] - 12.0f64.sqrt()).abs() <= 1e-15,
            "{w:?}"
        );
        assert!(
            rpa(vec![2.0, 12.0], vec![certified()]).omega().is_none(),
            "a witness outstanding beside an all-positive spectrum still withholds ω"
        );
        assert!(
            rpa(Vec::new(), Vec::new()).omega().is_none(),
            "an empty spectrum is None, never a vacuous Some(vec![])"
        );
        assert_eq!(
            rpa(Vec::new(), Vec::new()).omega2(),
            Some(Vec::new()),
            "…while Some(vec![]) IS the honest ω² answer for an empty channel"
        );
    }
}

#[cfg(test)]
mod t8f_bundle_tests {
    use super::SectorCtx;
    use crate::CoeffTensor;
    use crate::core::tangent::tangent_dim_grids;
    use crate::hamiltonian::SparseHamiltonian;
    use crate::pointgroup::Irrep;
    use crate::qn::{QnSpace, RankGrid};
    use crate::solvers::hess::weingarten_operator;
    use crate::solvers::riemannian::{normalize_mut, tangent_bases};
    use crate::test_util::{c2v_pg, randomized_state, seeded};
    use crate::vector::TtVector;

    #[allow(clippy::type_complexity)]
    fn pg_d4_channel() -> (
        QnSpace,
        CoeffTensor<f64>,
        CoeffTensor<f64>,
        SparseHamiltonian<f64>,
        TtVector<f64>,
        Irrep,
    ) {
        let pg = c2v_pg(4);
        let space = QnSpace::with_point_group(4, 2, 1, pg, Irrep::IDENTITY);
        let mut t = CoeffTensor::<f64>::zeros(4, 2);
        for p in 0..4 {
            t.set(&[p, p], 0.15 * (p as f64 - 2.0));
        }
        let mut v = CoeffTensor::<f64>::zeros(4, 4);
        for p in 0..4 {
            for q in 0..4 {
                if p != q {
                    v.set(&[p, p, q, q], 0.5 / (1.0 + (p as f64 - q as f64).abs()));
                }
            }
        }
        let ham = SparseHamiltonian::<f64>::new(space, &t, &v, None, f64::EPSILON);
        let mut rng = seeded(0x8F04);
        let mut x: TtVector<f64> = randomized_state(space, 1, 1, &mut rng);
        normalize_mut(&mut x);
        x.check();
        let (bl, _) = tangent_bases(&x);
        let phi = (1..pg.order())
            .map(Irrep::new)
            .find(|&phi| tangent_dim_grids(&space, &bl.ranks, phi) > 0)
            .expect(
                "fixture premise: c2v_pg(4) at (2,1) rank 1 must carry a LIVE shifted channel, or \
                 both arms below would run on an empty one and the bundle would never be built",
            );
        (space, t, v, ham, x, phi)
    }

    #[test]
    fn t8f_env_bundle_shapes_and_degenerate_workspaces() {
        let (space, t, v, ham, x, phi) = pg_d4_channel();
        let d = space.d;
        let space_out = space.with_target(space.target.fuse(phi));
        let h_out = SparseHamiltonian::<f64>::new(space_out, &t, &v, None, f64::EPSILON);
        let ctx = weingarten_operator(&ham, &x);
        let sctx = SectorCtx::new(&ctx, &h_out, phi);
        let (bl, _br) = sctx.bases();
        let r: Vec<RankGrid> = bl.ranks.clone();
        let r_col: Vec<RankGrid> = r.iter().map(|g| g.relabel_irrep(phi)).collect();
        println!(
            "t8f bundle: phi = {phi:?}, dim = {}, r = {:?}, r_col = {:?}",
            sctx.dim(),
            r.iter().map(RankGrid::total).collect::<Vec<_>>(),
            r_col.iter().map(RankGrid::total).collect::<Vec<_>>()
        );
        assert!(
            sctx.dim() > 0,
            "ANTI-VACUITY: the arm must run on a live channel"
        );
        assert_ne!(
            r, r_col,
            "ANTI-VACUITY: r and r_col must actually DIFFER on this channel, or the three-table \
             requirement (and every grid assertion below) is vacuous"
        );

        let e = sctx.envs();

        assert_eq!(e.u.space, space_out, "the left frame's stamp");
        assert_eq!(e.v.space, space_out, "the right frame's stamp");
        assert_eq!(e.u.ranks, r, "u.ranks must be base_l's own grid r");
        assert_eq!(
            e.v.ranks, r_col,
            "v.ranks must be the phi-RELABELED grid r_col, or the right stack is folded on the \
             wrong labels"
        );

        assert_eq!(e.l.len(), d, "the pure left stack has one entry per site");
        assert_eq!(e.r_phi.len(), d, "and so does the pure right stack");
        for k in 0..d {
            assert_eq!(e.l[k].k, k, "l[{k}] must be a sketch AT BOND {k}");
            assert_eq!(e.l[k].space, space_out, "l[{k}] must be stamped space_out");
            assert_eq!(
                e.r_phi[k].k,
                k + 1,
                "r_phi[{k}] must be a sketch AT BOND {}",
                k + 1
            );
            assert_eq!(
                e.r_phi[k].space, space_out,
                "r_phi[{k}] must be stamped space_out"
            );
            assert_eq!(
                e.l[k].row_ranks(),
                r[k],
                "l[{k}] rows must be r[{k}] element for element"
            );
            assert_eq!(
                e.l[k].col_ranks(),
                h_out.row_ranks(e.u.core(k)),
                "l[{k}] columns must be h_out's row-rank expansion of u_{k} — this is the axis \
                 that pairs with the ket at site {k}, and the axis on which an L summed into an \
                 L-tilde is a release-mode wrong answer"
            );
            assert_eq!(
                e.r_phi[k].col_ranks(),
                r_col[k + 1],
                "r_phi[{k}] columns must be r_col[{}] element for element",
                k + 1
            );
            assert_eq!(
                e.r_phi[k].row_ranks(),
                h_out.col_ranks(e.v.core(k)),
                "r_phi[{k}] rows must be h_out's col-rank expansion of v_{k}"
            );
        }
        assert_eq!(
            e.l[0].row_ranks(),
            r[0],
            "l[0] is the bond-0 identity seed at r[0]"
        );
        assert_eq!(
            e.r_phi[d - 1].col_ranks(),
            r_col[d],
            "r_phi[d-1] is the bond-d identity seed in SPACE_OUT at r_col[d] — a seed taken in \
             x's space would carry target Gamma0 and zero the stack with nothing asserted"
        );

        assert_eq!(e.ws_u.len(), d, "one ws_u per site");
        assert_eq!(e.ws_v.len(), d, "one ws_v per site");
        assert_eq!(e.ws_du.len(), d, "one ws_du per site");
        for k in 0..d {
            assert_eq!(
                e.ws_u[k].i_tab,
                h_out.row_ranges(e.u.core(k)),
                "ws_u[{k}] must be onesite_workspace(h_out, u_{k}) — the KET's tables"
            );
            assert_eq!(e.ws_u[k].j_tab, h_out.col_ranges(e.u.core(k)));
            assert_eq!(
                e.ws_v[k].i_tab,
                h_out.row_ranges(e.v.core(k)),
                "ws_v[{k}] must be onesite_workspace(h_out, v_{k})"
            );
            assert_eq!(e.ws_v[k].j_tab, h_out.col_ranges(e.v.core(k)));
            let decl = crate::core::SparseCore::<f64>::zeros(space_out, k, &r[k], &r_col[k + 1]);
            assert_eq!(
                e.ws_du[k].i_tab,
                h_out.row_ranges(&decl),
                "ws_du[{k}] must key on the DECLARED component shape r[{k}] x r_col[{}], not on \
                 any xi",
                k + 1
            );
            assert_eq!(e.ws_du[k].j_tab, h_out.col_ranges(&decl));
        }
        let du_last: usize = e.ws_u[d - 1]
            .j_tab
            .iter()
            .map(|(_, ranges)| ranges.iter().map(|q| q.end - q.start).sum::<usize>())
            .sum();
        let dv_first: usize = e.ws_v[0]
            .i_tab
            .iter()
            .map(|(_, ranges)| ranges.iter().map(|q| q.end - q.start).sum::<usize>())
            .sum();
        println!(
            "t8f bundle: degenerate tables — |ws_u[d-1].j_tab| = {du_last}, |ws_v[0].i_tab| = \
             {dv_first}"
        );
        assert_eq!(
            du_last, 0,
            "ws_u[d-1] is degenerate BY CONSTRUCTION: col_ranges names (N, N, space_out.target) \
             while u's bond-d mass sits at (N, N, Gamma0). It must be BUILT (it is, or the \
             indexing above would have panicked) and never asserted non-vacuous"
        );
        assert_eq!(
            dv_first, 0,
            "ws_v[0] is degenerate BY CONSTRUCTION: row_ranges names (0, 0, IDENTITY) while v's \
             bond-0 mass sits at (0, 0, phi)"
        );
    }

    #[test]
    fn t8f_bundle_is_lazy_and_built_once() {
        let (space, t, v, ham, x, phi) = pg_d4_channel();
        let space_out = space.with_target(space.target.fuse(phi));
        let h_out = SparseHamiltonian::<f64>::new(space_out, &t, &v, None, f64::EPSILON);
        let ctx = weingarten_operator(&ham, &x);
        let sctx = SectorCtx::new(&ctx, &h_out, phi);

        assert!(
            sctx.envs.get().is_none(),
            "SectorCtx::new must NOT build the environment bundle — an empty-channel context is \
             constructed only to report dim() = 0 and must not pay a two-sweep env build"
        );
        let basis = sctx.basis();
        assert!(!basis.is_empty(), "ANTI-VACUITY: the channel is live");
        assert!(
            sctx.envs.get().is_none(),
            "reading dim()/basis() must not force the bundle: {} {:?}",
            sctx.dim(),
            sctx.phi()
        );

        let _ = sctx.a_apply_fused(&basis[0]);
        let first = std::ptr::from_ref(
            sctx.envs
                .get()
                .expect("the first fused apply must build the bundle"),
        );

        let _ = sctx.a_apply_fused(&basis[0]);
        assert_eq!(
            std::ptr::from_ref(sctx.envs.get().expect("still built")),
            first,
            "a second fused apply must REUSE the bundle, not rebuild it"
        );
        let mut rng = seeded(0x8F05);
        let y: TtVector<f64> = randomized_state(space_out, 1, 2, &mut rng);
        let _ = sctx.project(&y);
        let _ = sctx.a_gram(&basis);
        let _ = sctx.grams(&basis);
        assert_eq!(
            std::ptr::from_ref(sctx.envs.get().expect("still built")),
            first,
            "project / a_gram / grams must share the ONE bundle a_apply_fused built"
        );
        println!(
            "t8f laziness: phi = {phi:?}, dim = {}, basis = {}, lambda = {}",
            sctx.dim(),
            basis.len(),
            sctx.lambda()
        );
    }
}

#[cfg(test)]
mod signed_tests;
