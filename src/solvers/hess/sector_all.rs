// Copyright (C) 2023-2026 Paul Cazeaux
//
// This file is part of qntt. qntt is free software: you can
// redistribute it and/or modify it under the terms of the GNU
// General Public License as published by the Free Software
// Foundation, either version 3 of the License, or (at your
// option) any later version. See LICENSE for details.

use std::time::Instant;

use crate::coeff::{CoeffTensor, ThreeBody};
use crate::core::tangent::{TtTangent, min_schmidt, tangent_basis, tangent_dim_grids};
use crate::hamiltonian::SparseHamiltonian;
use crate::integrals::{IntegralsLoad, PropertyOperator};
use crate::pointgroup::{Irrep, PointGroupKind};
use crate::solvers::joint::PropRoute;
use crate::vector::TtVector;

use super::channel::ChannelPlan;
use super::{
    ChannelLabel, IrrepSel, RpaModeKind, SectorCtx, SectorMomentOpts, SectorOpts, SectorReadout,
    SectorResponse, SectorStaticOpts, SectorStaticResponse, SectorTransitionMoments,
    SymplecticSpectrum, property_flux, symplectic_eigenvalues, symplectic_from_grams,
    weingarten_operator,
};

/// Which readout(s) to run per channel. `Both` is TWO independent solves reporting in two
/// different units — an energy and an `ω²` — never one solve read twice.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum SectorAllReadout {
    /// [`SectorCtx::sector_lowest_tda`] only: `eig(A^φ)`, an energy.
    Tda,
    /// [`SectorCtx::sector_lowest`] only: signed `ω²`.
    Rpa,
    /// Both, into [`SectorChannel::tda`] and [`SectorChannel::rpa`], from the SAME seed block.
    Both,
}

/// Which OBSERVABLES to evaluate per channel beside the spectral readout.
#[derive(Clone, Copy, Debug, Default, PartialEq, Eq)]
pub enum SectorObservables {
    /// Nothing: today's spectrum-only rows.
    #[default]
    None,
    /// The static `α^φ` only.
    Alpha,
    /// The transition moments, oscillator strengths and sum rules only.
    Moments,
    /// Both.
    Both,
}

/// Options of [`sector_response_all`].
#[derive(Clone, Debug)]
pub struct SectorAllOpts {
    /// Which irrep fluxes to enumerate. [`IrrepSel::All`] (the default) is `1..pg.order()`;
    /// [`IrrepSel::List`] preserves the caller's order and drops duplicates;
    /// [`IrrepSel::RefOnly`] enumerates NOTHING here.
    pub irreps: IrrepSel,
    /// Rendered irrep names on the labels; `None` leaves [`ChannelLabel::name`] unset. No reported
    /// number depends on it.
    pub kind: Option<PointGroupKind>,
    /// Which readout(s) to run per channel.
    pub readout: SectorAllReadout,
    /// Roots per channel per readout. `0` is REFUSED (see [`sector_response_all`], Panics).
    pub k: usize,
    /// Per-channel iterative tunables.
    pub solver: SectorOpts,
    /// Also run the DENSE certified oracle [`SectorCtx::symplectic_eigenvalues`] on any live
    /// channel with `dim <= this`. `None` = never.
    pub dense_cap: Option<usize>,
    /// Operator-construction drop threshold (`eps`) of every rebuilt `H(space_out)`. It is the
    /// operator's own coefficient threshold and has nothing to do with point-group screening — it
    /// does NOT remove a symmetry-forbidden 1e-14 residue.
    pub eps_h: f64,
    /// Also compute the in-sector (`φ = IDENTITY`) dense spectrum at `x` and store it in
    /// [`SectorSpectrumAll::reference_spectrum`]. Never a row in
    /// [`SectorSpectrumAll::channels`]; the reference METADATA is present either way.
    pub include_reference: bool,
    /// One-body observables to evaluate in every live channel.
    pub properties: Option<Vec<PropertyOperator<f64>>>,
    /// Which observables to compute per channel. Ignored when [`Self::properties`] is `None`.
    pub observables: SectorObservables,
}

impl Default for SectorAllOpts {
    fn default() -> Self {
        SectorAllOpts {
            irreps: IrrepSel::All,
            kind: None,
            readout: SectorAllReadout::Tda,
            k: 4,
            solver: SectorOpts::default(),
            dense_cap: None,
            eps_h: f64::EPSILON,
            include_reference: false,
            properties: None,
            observables: SectorObservables::None,
        }
    }
}

/// How a channel's `‖B^φ‖/‖A^φ‖` was scored. **The two arms are different numbers** — a Frobenius
/// norm ratio and a per-direction estimate on a subspace — and must never be printed in one column
/// without their tag: the campaign's acceptance criterion is a norm ratio, and quoting a seed
/// fraction as one would be a wrong number under a right name.
#[derive(Clone, Copy, Debug)]
pub enum BWeight {
    /// `‖B‖_F / ‖A‖_F` from the FULL Grams ([`SectorCtx::grams`], which returns `(A, B)` and not
    /// `(H^rr, H^ii)`). The honest norm ratio; available only where the dense route ran, since it
    /// is that route's Grams — scored from the same pair the Williamson reduction used, so it costs
    /// no extra apply. `f64::INFINITY` in the degenerate case `‖A‖_F == 0`, never a `0/0` NaN.
    GramRatio(f64),
    /// The per-direction Weingarten FRACTION `|⟨ξ, B^φ ξ⟩| / |⟨ξ, A^φ ξ⟩|` over the readout's own
    /// seed block — 2 applies per direction. An ESTIMATE on the seeds' subspace, not a norm ratio;
    /// `f64::INFINITY` for a direction whose A-form vanishes.
    SeedFraction { mean: f64, max: f64, n: usize },
}

/// What happened in one shifted channel. [`Self::Empty`] and the indefinite arm are first-class
/// OUTCOMES, not failures: a negative direction is information.
#[derive(Clone, Debug, PartialEq)]
pub enum SectorChannelStatus {
    /// Every requested root came back converged, no reported value was negative on EITHER produced
    /// arm, and the `Rpa` arm's solve (when it ran) met no indefinite metric and carries no
    /// signature −1 entry.
    Ok,
    /// `dim T^φ == 0`: the channel is structurally unreachable from this reference's grid.
    /// Detected BEFORE any operator is built (module header).
    Empty,
    /// Read out, and the block is INDEFINITE — `Γ0 ⊕ φ` holds a state below `λ`, or the `Rpa`
    /// pencil left the definite case.
    Indefinite {
        n_negative: usize,
        certified: bool,
        n_negative_signature: usize,
        metric_indefinite: bool,
    },
    /// The spectrum read out, but an OBSERVABLE was refused — a genuine `Err` from
    /// `SectorCtx::polarizability_static` / `transition_moments_dense` (`NotTtCritical`,
    /// `HessianIndefinite`, `SolveNotConverged`, `SymplecticUnstable`, …). The row's
    /// [`SectorChannel::alpha`] / [`SectorChannel::moments`] are then `None`, and this is where the
    /// reason lives rather than being dropped. ★ A `Moments` request on a channel
    /// [`SectorAllOpts::dense_cap`] did NOT admit is "not attempted", not a refusal: `moments`
    /// stays `None` and the status is unchanged (see [`SectorChannel::moments`]).
    ObservableRefused { what: &'static str, why: String },
    /// A requested readout came back unconverged, or short of `k.min(dim)` roots — as often a SPAN
    /// statement at a truncated base point as a tolerance failure
    /// ([`SectorResponse::spectrum`]'s doc).
    NotConverged,
}

/// One shifted channel's row.
#[derive(Clone, Debug)]
pub struct SectorChannel {
    /// `ChannelLabel { irrep: Γ0 ⊕ φ, name, spin_delta: 0 }` — the SHIPPED label type, so both
    /// composition drivers print channel names the same way. `spin_delta` is `0`: an irrep channel
    /// carries no spin offset.
    pub label: ChannelLabel,
    /// The channel's irrep FLUX.
    pub phi: Irrep,
    /// `Γ0 ⊕ φ`, i.e. `label.irrep` — kept as its own field because the code reads it as a target
    /// far more often than as a name.
    pub target: Irrep,
    /// `dim T^φ` ([`tangent_dim_grids`]), STRUCTURAL. **Not the dimension of the `Γ0 ⊕ φ`
    /// sector** — module header.
    pub dim: usize,
    /// The `Tda` readout when it was requested.
    pub tda: Option<SectorResponse>,
    /// The `Rpa` readout when it was requested.
    pub rpa: Option<SectorResponse>,
    /// The dense certified Williamson spectrum when [`SectorAllOpts::dense_cap`] admitted this
    /// channel.
    pub dense: Option<SymplecticSpectrum>,
    /// Weingarten weight of the channel, and how it was scored. `None` on an `Empty` row (there
    /// is no `A^φ` to score against) and on a live row whose whole seed block was annihilated.
    pub b_weight: Option<BWeight>,
    /// The channel's static polarizability when [`SectorAllOpts::observables`] asked for it and
    /// it was produced. `None` on an [`Empty`](SectorChannelStatus::Empty) row, when it was not
    /// requested, and when it was REFUSED — in which case the reason is in [`Self::status`].
    pub alpha: Option<SectorStaticResponse>,
    /// The channel's transition moments, on the same terms as [`Self::alpha`] plus the
    /// [`SectorAllOpts::dense_cap`] gate: the moment route is the DENSE one, so on a live channel
    /// with `dim > dense_cap` (or `dense_cap = None`) it is NOT ATTEMPTED — `None` here with the
    /// status UNCHANGED, because declining a cost the caller capped is not a refusal and
    /// [`SectorChannelStatus::ObservableRefused`] is reserved for a genuine `Err`. Tell the two
    /// apart by [`Self::dense`]: `Some` means the cap admitted the channel, so a `None` here beside
    /// it is a refusal recorded in [`Self::status`].
    pub moments: Option<SectorTransitionMoments>,
    /// The join of the channel's outcomes.
    pub status: SectorChannelStatus,
    /// Wall of this channel, the operator build INCLUDED — at molecular rank that build is the
    /// dominant per-channel cost, and hiding it would make the readout look free.
    pub secs: f64,
}

/// The reference point's metadata. **Not a spectrum** — the in-sector question is not this
/// driver's (module header); [`SectorSpectrumAll::reference_spectrum`] is the opt-in field.
#[derive(Clone, Copy, Debug)]
pub struct SectorReference {
    /// The Rayleigh quotient at `x` — the shift of every `A^φ = P^φ(H_out − λ)P^φ`.
    pub lambda: f64,
    /// `Γ0`, the reference's own irrep.
    pub target: Irrep,
    /// `tangent_dim(x)` — identically `tangent_dim_grids(&space, &ranks, IDENTITY)`, the two
    /// spellings of the same count — for the `Σ_φ dim T^φ + tangent_dim(x) =
    /// tangent_dim(x.forget_point_group())` identity ([`SectorSpectrumAll::dim_total`]).
    pub tangent_dim: usize,
    /// Smallest Schmidt value at the base point ([`min_schmidt`]) — base-point conditioning,
    /// quoted with every reading.
    pub sigma_min: f64,
}

/// The composed, labeled cross-sector spectrum.
#[derive(Clone, Debug)]
pub struct SectorSpectrumAll {
    /// The reference point's metadata, always present.
    pub reference: SectorReference,
    /// The in-sector dense spectrum at `x`, present exactly when
    /// [`SectorAllOpts::include_reference`] was set. Never a row of [`Self::channels`]: it is a
    /// different operator with a different deflation and a different result type (module header).
    pub reference_spectrum: Option<SymplecticSpectrum>,
    /// Enumeration order: ascending irrep index over `1..pg.order()`, or the caller's
    /// [`IrrepSel::List`] order. `IDENTITY` never appears.
    pub channels: Vec<SectorChannel>,
    /// Channel Hamiltonians actually CONSTRUCTED. An [`SectorChannelStatus::Empty`] row builds
    /// none, so this is the count of live channels — the exact statement the per-row `secs` can
    /// only approximate.
    pub built: usize,
    /// `Σ_channels α^φ`, row-major over the components the caller passed, with the per-channel
    /// table kept on the rows. Present exactly when [`SectorAllOpts::properties`] was supplied and
    /// [`SectorAllOpts::observables`] asked for α.
    pub alpha_total: Option<Vec<f64>>,
    /// `Σ_channels Σ_n f_n` — the cross-channel oscillator-strength total. Present exactly when
    /// moments were requested and produced.
    pub oscillator_total: Option<f64>,
    /// Labels of the properties computed NOWHERE, in the caller's order: those whose derived flux
    /// is [`Irrep::IDENTITY`] (the in-sector problem's) and those carrying more than one flux
    /// class. Reported by name rather than dropped.
    pub skipped: Vec<String>,
}

/// One level of the merged spectrum, in its readout's OWN units.
#[derive(Clone, Copy, Debug)]
pub struct SectorLevel {
    /// `λ + value` for a `Tda` level; `λ + √ω²` for a NON-NEGATIVE `Rpa` level of signature `+1`
    /// from an UNFLAGGED solve; `f64::NAN` for a negative `ω²` (a resonance `ω = i√|ω²|` that no
    /// real energy can carry), for an entry of Krein signature −1 (a de-excitation to a state at
    /// `λ − |a|`, which `λ + √ω²` would put at `λ + |a|`) and for EVERY entry of a solve flagged
    /// metric-indefinite (its `converged` does not license an `ω`, Q5 — read the per-mode checks).
    pub energy: f64,
    /// The raw readout value in its OWN units: `eig(A^φ)` (an energy) on `Tda`, signed `ω²` on
    /// `Rpa`. Read [`Self::readout`] before reading this.
    pub value: f64,
    /// Which readout produced it.
    pub readout: SectorReadout,
    /// Index into [`SectorSpectrumAll::channels`].
    pub channel: usize,
    /// `j`-th root of that channel's readout.
    pub k: usize,
}

impl SectorSpectrumAll {
    /// Every channel's `readout` values merged, sorted ascending by [`SectorLevel::value`], ties
    /// broken by `(channel, k)`.
    pub fn levels(&self, readout: SectorReadout) -> Vec<SectorLevel> {
        let lambda = self.reference.lambda;
        let mut out: Vec<SectorLevel> = Vec::new();
        for (ci, ch) in self.channels.iter().enumerate() {
            let arm = match readout {
                SectorReadout::Tda => ch.tda.as_ref(),
                SectorReadout::Rpa => ch.rpa.as_ref(),
            };
            let vals: Vec<f64> = match readout {
                SectorReadout::Tda => arm.and_then(SectorResponse::theta).map(<[f64]>::to_vec),
                SectorReadout::Rpa => arm.and_then(SectorResponse::omega2),
            }
            .unwrap_or_default();
            let flagged = arm.is_some_and(SectorResponse::metric_indefinite);
            for (k, &value) in vals.iter().enumerate() {
                let de_excitation = arm
                    .and_then(SectorResponse::rpa)
                    .and_then(|r| r.modes.get(k))
                    .is_some_and(|m| m.kind == RpaModeKind::Real && m.krein < 0.0);
                let energy = match readout {
                    SectorReadout::Tda => lambda + value,
                    SectorReadout::Rpa if value >= 0.0 && !flagged && !de_excitation => {
                        lambda + value.sqrt()
                    }
                    SectorReadout::Rpa => f64::NAN,
                };
                out.push(SectorLevel {
                    energy,
                    value,
                    readout,
                    channel: ci,
                    k,
                });
            }
        }
        out.sort_by(|a, b| {
            a.value
                .total_cmp(&b.value)
                .then(a.channel.cmp(&b.channel))
                .then(a.k.cmp(&b.k))
        });
        out
    }

    /// The channels with [`SectorChannelStatus::Empty`] — the "this irrep is unreachable from this
    /// reference" line every consumer wants.
    pub fn empty_channels(&self) -> impl Iterator<Item = &SectorChannel> {
        self.channels
            .iter()
            .filter(|c| c.status == SectorChannelStatus::Empty)
    }

    /// `Σ_φ dim` over the enumerated channels plus [`SectorReference::tangent_dim`].
    pub fn dim_total(&self) -> usize {
        self.channels.iter().map(|c| c.dim).sum::<usize>() + self.reference.tangent_dim
    }
}

fn enumerate_fluxes(order: u8, sel: &IrrepSel) -> Vec<Irrep> {
    match sel {
        IrrepSel::RefOnly => Vec::new(),
        IrrepSel::All => (1..order).map(Irrep::new).collect(),
        IrrepSel::List(list) => {
            let mut out: Vec<Irrep> = Vec::new();
            for &ir in list {
                if ir != Irrep::IDENTITY && ir.index() < order && !out.contains(&ir) {
                    out.push(ir);
                }
            }
            out
        }
    }
}

fn classify(
    nroots: usize,
    tda: Option<&SectorResponse>,
    rpa: Option<&SectorResponse>,
    observable: Option<&(&'static str, String)>,
) -> SectorChannelStatus {
    let negatives = |r: &SectorResponse| {
        r.theta()
            .map(<[f64]>::to_vec)
            .or_else(|| r.omega2())
            .unwrap_or_default()
            .iter()
            .filter(|z| **z < 0.0)
            .count()
    };
    let n_tda = tda.map_or(0, negatives);
    let n_rpa = rpa.map_or(0, negatives);
    let n_sig = rpa.and_then(SectorResponse::rpa).map_or(0, |r| {
        r.modes
            .iter()
            .filter(|m| m.kind == RpaModeKind::Real && m.krein < 0.0)
            .count()
    });
    let flagged = rpa.is_some_and(SectorResponse::metric_indefinite);
    if n_tda > 0 || n_rpa > 0 || n_sig > 0 || flagged {
        let (n_negative, witness) = if n_tda > 0 {
            (n_tda, tda)
        } else {
            (n_rpa, rpa)
        };
        return SectorChannelStatus::Indefinite {
            n_negative,
            certified: witness
                .expect("the arm that supplied the verdict ran")
                .all_negatives_certified(),
            n_negative_signature: n_sig,
            metric_indefinite: flagged,
        };
    }
    if let Some((what, why)) = observable {
        return SectorChannelStatus::ObservableRefused {
            what,
            why: why.clone(),
        };
    }
    let short_or_stalled = |r: &SectorResponse| {
        let n = r
            .theta()
            .map(<[f64]>::len)
            .or_else(|| r.omega2().map(|w| w.len()))
            .unwrap_or(0);
        !r.converged || n < nroots
    };
    if tda.is_some_and(short_or_stalled) || rpa.is_some_and(short_or_stalled) {
        return SectorChannelStatus::NotConverged;
    }
    SectorChannelStatus::Ok
}

fn seed_fraction(sc: &SectorCtx<'_, '_, f64>, n_seeds: usize, seed: u64) -> Option<BWeight> {
    let seeds = sc.seed_block(n_seeds, seed);
    if seeds.is_empty() {
        return None;
    }
    let mut sum = 0.0f64;
    let mut max = 0.0f64;
    for xi in &seeds {
        let fa = xi.dot(&sc.a_apply(xi)).abs();
        let fb = xi.dot(&sc.b_apply(xi)).abs();
        let frac = if fa == 0.0 { f64::INFINITY } else { fb / fa };
        sum += frac;
        max = max.max(frac);
    }
    Some(BWeight::SeedFraction {
        mean: sum / seeds.len() as f64,
        max,
        n: seeds.len(),
    })
}

/// Cross-sector linear response in EVERY enumerated irrep channel at one typed reference: build
/// and drop one operator per channel, read it out, merge the rows.
///
/// # Panics
///
/// - `k == 0`. Both kernels treat `nroots == 0` as "nothing was requested, nothing is
///   outstanding" and report `converged = true` beside a POSITIVE `dim` — the one case where
///   `converged` does not mean what a channel-LOOPING caller reads it to mean. This function IS
///   that looping caller, so it refuses rather than compose a table of vacuous convergences.
/// - `h.space != x.space` (the reference operator is not the reference point's).
pub fn sector_response_all(
    h: &SparseHamiltonian<f64>,
    x: &TtVector<f64>,
    t: &CoeffTensor<f64>,
    v: &CoeffTensor<f64>,
    w: Option<&dyn ThreeBody<f64>>,
    opts: &SectorAllOpts,
) -> SectorSpectrumAll {
    assert!(
        opts.k >= 1,
        "sector_response_all: k == 0 would ask every channel for no roots at all, and both sector \
         readouts report that as `converged = true` beside a positive dim — a table of vacuous \
         convergences. Pass k >= 1"
    );
    assert_eq!(
        h.space, x.space,
        "sector_response_all: the reference operator must be built on the reference point's own \
         space"
    );

    let ctx = weingarten_operator(h, x);
    let (base_l, base_r) = ctx.bases();
    let space = base_l.space;
    let ranks = base_l.ranks.clone();
    let reference = SectorReference {
        lambda: ctx.lambda,
        target: space.target,
        tangent_dim: tangent_dim_grids(&space, &ranks, Irrep::IDENTITY),
        sigma_min: min_schmidt(base_l, base_r),
    };
    let reference_spectrum = opts.include_reference.then(|| {
        let basis = tangent_basis(base_l, base_r, true);
        symplectic_eigenvalues(&ctx, &basis)
    });

    let mut skipped: Vec<String> = Vec::new();
    let props_zeroed: Option<Vec<PropertyOperator<f64>>> = opts.properties.as_ref().map(|props| {
        assert!(
            props.len() <= 3,
            "sector_response_all: at most three Cartesian components ({} given) — \
             `oscillator_strengths` is a three-row kernel whose 2/3 is the isotropic average over \
             x, y, z",
            props.len()
        );
        props
            .iter()
            .map(|p| match property_flux(p, space.pg) {
                Ok(Some(g)) if g != Irrep::IDENTITY => p.clone(),
                other => {
                    if !matches!(other, Ok(None)) {
                        skipped.push(p.label.clone());
                    }
                    PropertyOperator::new(p.label.clone(), CoeffTensor::<f64>::zeros(space.d, 2))
                }
            })
            .collect()
    });
    let n_props = props_zeroed.as_ref().map_or(0, Vec::len);
    let want_alpha = matches!(
        opts.observables,
        SectorObservables::Alpha | SectorObservables::Both
    );
    let want_moments = matches!(
        opts.observables,
        SectorObservables::Moments | SectorObservables::Both
    );
    let mut alpha_total =
        (props_zeroed.is_some() && want_alpha).then(|| vec![0.0f64; n_props * n_props]);
    let mut oscillator_total = (props_zeroed.is_some() && want_moments).then_some(0.0f64);

    let mut channels: Vec<SectorChannel> = Vec::new();
    let mut built = 0usize;
    for phi in enumerate_fluxes(space.pg.order(), &opts.irreps) {
        let t0 = Instant::now();
        let target = space.target.fuse(phi);
        let label = ChannelPlan {
            spin_delta: 0,
            irrep: target,
        }
        .label(opts.kind);
        let dim = tangent_dim_grids(&space, &ranks, phi);
        if dim == 0 {
            channels.push(SectorChannel {
                label,
                phi,
                target,
                dim: 0,
                tda: None,
                rpa: None,
                dense: None,
                b_weight: None,
                alpha: None,
                moments: None,
                status: SectorChannelStatus::Empty,
                secs: t0.elapsed().as_secs_f64(),
            });
            continue;
        }

        let space_out = space.with_target(target);
        let h_out = SparseHamiltonian::<f64>::new(space_out, t, v, w, opts.eps_h);
        built += 1;
        let sc = SectorCtx::new(&ctx, &h_out, phi);

        let arm = |readout: SectorReadout| SectorOpts {
            readout,
            ..opts.solver.clone()
        };
        let tda = matches!(opts.readout, SectorAllReadout::Tda | SectorAllReadout::Both)
            .then(|| sc.sector_lowest_tda(opts.k, &arm(SectorReadout::Tda)));
        let rpa = matches!(opts.readout, SectorAllReadout::Rpa | SectorAllReadout::Both)
            .then(|| sc.sector_lowest(opts.k, &arm(SectorReadout::Rpa)));

        let dense_grams: Option<DenseGrams> =
            opts.dense_cap.is_some_and(|cap| dim <= cap).then(|| {
                let basis = sc.basis();
                let (a, b) = sc.grams(&basis);
                (basis, a, b)
            });
        let (dense, b_weight) = match dense_grams.as_ref() {
            Some((basis, a, b)) => {
                let na = frobenius(a);
                let ratio = if na == 0.0 {
                    f64::INFINITY
                } else {
                    frobenius(b) / na
                };
                (
                    Some(symplectic_from_grams(a, b, basis.len(), 1e-10)),
                    Some(BWeight::GramRatio(ratio)),
                )
            }
            None => (
                None,
                seed_fraction(&sc, opts.k.min(dim) + 2, opts.solver.seed),
            ),
        };

        let mut obs_refusal: Option<(&'static str, String)> = None;
        let mut alpha: Option<SectorStaticResponse> = None;
        let mut moments: Option<SectorTransitionMoments> = None;
        if let Some(pz) = props_zeroed.as_ref() {
            match sc.covectors(pz, "sector_response_all") {
                Err(e) => {
                    let what = if want_alpha { "alpha" } else { "moments" };
                    obs_refusal = Some((what, e.to_string()));
                }
                Ok(covs) => {
                    if want_alpha {
                        match sc.polarizability_static_from_covectors(
                            pz,
                            &covs,
                            &SectorStaticOpts::default(),
                        ) {
                            Ok(r) => {
                                if let Some(total) = alpha_total.as_mut() {
                                    for (k, z) in r.alpha.iter().enumerate() {
                                        total[k] += z;
                                    }
                                }
                                alpha = Some(r);
                            }
                            Err(e) => obs_refusal = Some(("alpha", e.to_string())),
                        }
                    }
                    if let (true, Some((basis, a, b))) = (want_moments, dense_grams.as_ref()) {
                        let mopts = SectorMomentOpts {
                            solver: opts.solver.clone(),
                            route: PropRoute::Materialized,
                            ..SectorMomentOpts::default()
                        };
                        match sc.transition_moments_from_grams(pz, &covs, basis, a, b, &mopts) {
                            Ok(m) => {
                                if let Some(total) = oscillator_total.as_mut() {
                                    *total += m.oscillator.iter().sum::<f64>();
                                }
                                moments = Some(m);
                            }
                            Err(e) => {
                                obs_refusal.get_or_insert(("moments", e.to_string()));
                            }
                        }
                    }
                }
            }
        }

        let status = classify(
            opts.k.min(dim),
            tda.as_ref(),
            rpa.as_ref(),
            obs_refusal.as_ref(),
        );
        channels.push(SectorChannel {
            label,
            phi,
            target,
            dim,
            tda,
            rpa,
            dense,
            b_weight,
            alpha,
            moments,
            status,
            secs: t0.elapsed().as_secs_f64(),
        });
    }

    SectorSpectrumAll {
        reference,
        reference_spectrum,
        channels,
        built,
        alpha_total,
        oscillator_total,
        skipped,
    }
}

type DenseGrams = (Vec<TtTangent<f64>>, Vec<f64>, Vec<f64>);

fn frobenius(g: &[f64]) -> f64 {
    g.iter().map(|z| z * z).sum::<f64>().sqrt()
}

/// [`sector_response_all`] from a loaded integrals bundle — SCREENED HERE, unconditionally, and
/// the recommended entry.
///
/// # Panics
///
/// Where [`sector_response_all`] does, plus: if `x.space` is not on the load's lattice, or if the
/// load's integrals carry a symmetry-forbidden entry above
/// [`crate::fcidump::SYMMETRY_FORBIDDEN_TOL`] — that is a genuine mismatch between the
/// coefficients and the resolved group, not roundoff, and the screen REFUSES it rather than
/// silently zeroing it.
pub fn sector_response_all_from_load(
    load: &IntegralsLoad<f64>,
    x: &TtVector<f64>,
    opts: &SectorAllOpts,
) -> SectorSpectrumAll {
    assert!(
        x.space.same_lattice(&load.space),
        "sector_response_all_from_load: the reference point lives on a different lattice from the \
         load ({:?} vs {:?}) — d, nup, ndn and the point group (kind and the ORBSYM table) must \
         agree; `target` deliberately is not compared",
        x.space,
        load.space
    );
    let mut opts = opts.clone();
    if opts.kind.is_none() {
        opts.kind = Some(load.pg.kind);
    }
    let w: Option<&dyn ThreeBody<f64>> = load.w.as_ref().map(|w| w as &dyn ThreeBody<f64>);
    if load.space.pg.is_trivial() {
        let h = SparseHamiltonian::<f64>::new(x.space, &load.ints.t, &load.ints.v, w, opts.eps_h);
        return sector_response_all(&h, x, &load.ints.t, &load.ints.v, w, &opts);
    }
    let mut t = load.ints.t.clone();
    let mut v = load.ints.v.clone();
    crate::fcidump::screen_point_group_forbidden(&mut t, &mut v, load.space.pg).unwrap_or_else(
        |e| {
            panic!(
                "sector_response_all_from_load: the loaded integrals are not compatible with the \
                 resolved point group ({e}). A ~1e-14 forbidden residue is screened silently; \
                 this entry is above the refusal tolerance, so the coefficients and the group \
                 disagree — check the dump's ORBSYM or the PointGroupPolicy."
            )
        },
    );
    let h = SparseHamiltonian::<f64>::new(x.space, &t, &v, w, opts.eps_h);
    sector_response_all(&h, x, &t, &v, w, &opts)
}

#[cfg(test)]
mod tests {

    use super::*;
    use crate::scalar::C64;
    use crate::solvers::hess::{
        MetricEvent, NegativeWitness, RpaMetric, RpaMode, RpaReadout, RpaReport, SectorSpectrum,
    };

    fn rpa(
        omega2: Vec<f64>,
        converged: bool,
        applies: usize,
        negative: Vec<NegativeWitness>,
    ) -> SectorResponse {
        let n = omega2.len();
        signed(omega2, vec![1; n], converged, applies, negative, None)
    }

    fn signed(
        omega2: Vec<f64>,
        signature: Vec<i8>,
        converged: bool,
        applies: usize,
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
        let (events, n_negative_basis) = report.map_or((Vec::new(), 0), |r| (r.events, r.held));
        SectorResponse {
            spectrum: SectorSpectrum::Rpa(RpaReadout {
                passed_over: Vec::new(),
                modes,
                converged,
                iterations: 3,
                applies,
                report: RpaReport::Signed {
                    metric: RpaMetric::Rr,
                    events,
                    n_negative_basis,
                    general_path: false,
                    diag: None,
                },
            }),
            dim: 4,
            phi: Irrep::new(1),
            target: Irrep::new(1),
            converged,
            iterations: 3,
            applies,
            sigma_min: 0.5,
        }
    }

    fn tda(theta: Vec<f64>, converged: bool, negative: Vec<NegativeWitness>) -> SectorResponse {
        SectorResponse {
            spectrum: SectorSpectrum::Tda { theta, negative },
            dim: 4,
            phi: Irrep::new(1),
            target: Irrep::new(1),
            converged,
            iterations: 3,
            applies: 6,
            sigma_min: 0.5,
        }
    }

    fn witness(index: usize, theta: f64, fresh: f64) -> NegativeWitness {
        NegativeWitness {
            index,
            theta,
            fresh,
            fresh_w: 1.0,
        }
    }

    struct Report {
        events: Vec<MetricEvent>,
        held: usize,
    }

    fn report(held: usize) -> Report {
        let ev = |accepted: bool| MetricEvent {
            basis_len: 2,
            gn_before: 1.0,
            gn_after: 0.8,
            wnorm2: if accepted { -0.3 } else { -2e-7 },
            accepted,
        };
        Report {
            events: if held == 0 {
                vec![ev(false)]
            } else {
                (0..held).map(|_| ev(true)).collect()
            },
            held,
        }
    }

    fn refused() -> (&'static str, String) {
        ("alpha", "why".to_string())
    }

    #[test]
    fn classify_join_precedence_is_indefinite_refused_not_converged_ok() {
        assert_eq!(
            classify(2, None, Some(&rpa(vec![0.1, 0.2], true, 5, vec![])), None),
            SectorChannelStatus::Ok
        );
        assert_eq!(
            classify(2, None, Some(&rpa(vec![0.1, 0.2], false, 5, vec![])), None),
            SectorChannelStatus::NotConverged
        );
        assert_eq!(
            classify(2, Some(&tda(vec![0.1], true, vec![])), None, None),
            SectorChannelStatus::NotConverged
        );
        let r = refused();
        assert_eq!(
            classify(2, None, Some(&rpa(vec![0.1], false, 5, vec![])), Some(&r)),
            SectorChannelStatus::ObservableRefused {
                what: "alpha",
                why: "why".to_string()
            }
        );
        assert_eq!(
            classify(
                2,
                None,
                Some(&rpa(vec![0.1, 0.2], true, 5, vec![])),
                Some(&r)
            ),
            SectorChannelStatus::ObservableRefused {
                what: "alpha",
                why: "why".to_string()
            }
        );
        assert_eq!(
            classify(
                2,
                None,
                Some(&rpa(
                    vec![-0.3, 0.2],
                    true,
                    5,
                    vec![witness(0, -0.3, -0.29)]
                )),
                Some(&r)
            ),
            SectorChannelStatus::Indefinite {
                n_negative: 1,
                certified: true,
                n_negative_signature: 0,
                metric_indefinite: false
            }
        );
        assert_eq!(
            classify(
                2,
                None,
                Some(&rpa(vec![-0.3, 0.2], true, 5, vec![witness(0, -0.3, 0.01)])),
                Some(&r)
            ),
            SectorChannelStatus::Indefinite {
                n_negative: 1,
                certified: false,
                n_negative_signature: 0,
                metric_indefinite: false
            }
        );
        assert_eq!(
            classify(
                2,
                Some(&tda(vec![0.4, 0.5], true, vec![])),
                Some(&rpa(vec![-0.3, 0.2], true, 5, vec![witness(0, -0.3, -0.2)])),
                Some(&r)
            ),
            SectorChannelStatus::Indefinite {
                n_negative: 1,
                certified: true,
                n_negative_signature: 0,
                metric_indefinite: false
            }
        );
        assert_eq!(
            classify(
                2,
                Some(&tda(
                    vec![-0.4, -0.1],
                    true,
                    vec![witness(0, -0.4, -0.4), witness(1, -0.1, -0.1)]
                )),
                Some(&rpa(vec![-0.3, 0.2], true, 5, vec![witness(0, -0.3, -0.2)])),
                None
            ),
            SectorChannelStatus::Indefinite {
                n_negative: 2,
                certified: true,
                n_negative_signature: 0,
                metric_indefinite: false
            }
        );
        assert_eq!(
            classify(
                2,
                Some(&tda(vec![-0.4], true, vec![witness(0, -0.4, -0.4)])),
                Some(&rpa(Vec::new(), false, 9, vec![])),
                Some(&r)
            ),
            SectorChannelStatus::Indefinite {
                n_negative: 1,
                certified: true,
                n_negative_signature: 0,
                metric_indefinite: false
            }
        );
        assert_eq!(
            classify(2, None, Some(&rpa(Vec::new(), false, 9, vec![])), None),
            SectorChannelStatus::NotConverged
        );
        assert_eq!(
            classify(
                2,
                None,
                Some(&signed(
                    vec![0.1, 0.2],
                    vec![-1, 1],
                    true,
                    5,
                    vec![],
                    Some(report(1))
                )),
                Some(&r)
            ),
            SectorChannelStatus::Indefinite {
                n_negative: 0,
                certified: true,
                n_negative_signature: 1,
                metric_indefinite: true
            }
        );
        assert_eq!(
            classify(
                2,
                None,
                Some(&signed(
                    vec![0.1, 0.2],
                    vec![1, 1],
                    true,
                    5,
                    vec![],
                    Some(report(0))
                )),
                Some(&r)
            ),
            SectorChannelStatus::Indefinite {
                n_negative: 0,
                certified: true,
                n_negative_signature: 0,
                metric_indefinite: true
            }
        );
        assert_eq!(
            classify(2, None, Some(&rpa(Vec::new(), false, 0, vec![])), None),
            SectorChannelStatus::NotConverged
        );
    }

    fn channel(
        phi: u8,
        tda_arm: Option<SectorResponse>,
        rpa_arm: Option<SectorResponse>,
    ) -> SectorChannel {
        SectorChannel {
            label: ChannelLabel {
                irrep: Irrep::new(phi),
                name: None,
                spin_delta: 0,
            },
            phi: Irrep::new(phi),
            target: Irrep::new(phi),
            dim: 4,
            tda: tda_arm,
            rpa: rpa_arm,
            dense: None,
            b_weight: None,
            alpha: None,
            moments: None,
            status: SectorChannelStatus::Ok,
            secs: 0.0,
        }
    }

    #[test]
    fn levels_withhold_the_energy_of_a_flagged_or_negative_signature_entry() {
        let lambda = -1.0f64;
        let spec = SectorSpectrumAll {
            reference: SectorReference {
                lambda,
                target: Irrep::IDENTITY,
                tangent_dim: 5,
                sigma_min: 0.5,
            },
            reference_spectrum: None,
            channels: vec![
                channel(1, None, Some(rpa(vec![0.25, 1.0], true, 5, vec![]))),
                channel(
                    2,
                    Some(tda(vec![0.3], true, vec![])),
                    Some(signed(
                        vec![0.36, 4.0],
                        vec![1, 1],
                        true,
                        5,
                        vec![],
                        Some(report(0)),
                    )),
                ),
                channel(
                    3,
                    None,
                    Some(signed(
                        vec![-0.5, 0.09, 0.49],
                        vec![-1, -1, 1],
                        true,
                        5,
                        vec![witness(0, -0.5, -0.5)],
                        None,
                    )),
                ),
            ],
            built: 3,
            alpha_total: None,
            oscillator_total: None,
            skipped: Vec::new(),
        };
        let lv = spec.levels(SectorReadout::Rpa);
        assert_eq!(
            lv.len(),
            7,
            "every omega^2 entry stays a level, published or not"
        );
        let find = |c: usize, k: usize| {
            *lv.iter()
                .find(|l| l.channel == c && l.k == k)
                .unwrap_or_else(|| panic!("level ({c}, {k}) missing"))
        };
        let close = |a: f64, b: f64| (a - b).abs() <= 1e-15;
        assert!(close(find(0, 0).energy, lambda + 0.5) && close(find(0, 1).energy, lambda + 1.0));
        for k in 0..2 {
            assert!(
                find(1, k).energy.is_nan(),
                "flagged solve: entry {k} must carry no energy"
            );
        }
        assert!(
            find(2, 0).energy.is_nan(),
            "a negative omega^2 is a resonance"
        );
        assert!(
            find(2, 1).energy.is_nan(),
            "an s = -1 entry is a de-excitation: lambda + |a| would be the wrong state"
        );
        assert!(
            close(find(2, 2).energy, lambda + 0.7),
            "the s = +1 entry is published"
        );
        assert!(
            close(find(1, 1).value, 4.0) && close(find(2, 1).value, 0.09),
            "value keeps the raw omega^2"
        );
        let lt = spec.levels(SectorReadout::Tda);
        assert_eq!(lt.len(), 1);
        assert!(
            close(lt[0].energy, lambda + 0.3),
            "the Tda merge is untouched by the Rpa arm's flag"
        );
    }
}
