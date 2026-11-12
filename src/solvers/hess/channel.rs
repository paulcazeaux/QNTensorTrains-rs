// Copyright (C) 2023-2026 Paul Cazeaux
//
// This file is part of qntt. qntt is free software: you can
// redistribute it and/or modify it under the terms of the GNU
// General Public License as published by the Free Software
// Foundation, either version 3 of the License, or (at your
// option) any later version. See LICENSE for details.

use crate::coeff::CoeffTensor;
use crate::hamiltonian::{SparseHamiltonian, rayleigh_quotient};
use crate::integrals::IntegralsLoad;
use crate::pointgroup::{Irrep, PointGroupKind};
use crate::qn::{PerSector, Pooled, QnSpace, RankGrid};
use crate::round::RoundMode;
use crate::solvers::mals::{MalsOptions, NoiseType, mals};
use crate::solvers::riemannian::{RiemannianOpts, exact_gradnorm, normalize_mut, riemannian_cg};
use crate::vector::{TtVector, tt_randn, tt_state_in};
use rand::SeedableRng;
use rand_xoshiro::Xoshiro256PlusPlus;

use super::{ConeOpts, ConeResponse, try_cone_response};

/// Which point-group irreps to enumerate as channels.
#[derive(Clone, Debug, PartialEq, Eq)]
pub enum IrrepSel {
    /// Only the base space's own target irrep.
    RefOnly,
    /// Every irrep in `0..pg.order()`; the ones whose typed space is `max_ranks`-empty are
    /// reported as [`ChannelStatus::Empty`] without ever building an `H`.
    All,
    /// A caller-supplied list, order preserved, duplicates removed.
    List(Vec<Irrep>),
}

/// Which spin channels to enumerate, as an offset from the base space's own spin sector.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum SpinSel {
    /// Only the base space's own spin sector.
    RefOnly,
    /// `delta = 1..=m`: abelian `(nup + delta, ndn − delta)`, SU(2) `two_s + 2·delta`. The
    /// sector-degeneracy trick used CONSTRUCTIVELY — an `S ≥ Sz + delta` multiplet's lowest
    /// member is that channel's ground state, and those are exactly the tier-1 states linear
    /// response cannot reach. Infeasible parameters are reported as [`ChannelStatus::Empty`].
    Raised(usize),
}

/// The channel set: the cartesian product of an irrep selection and a spin selection, with the
/// reference channel `(spin_delta = 0, irrep = base.target)` always enumerated FIRST.
#[derive(Clone, Debug)]
pub struct ChannelSet {
    pub irreps: IrrepSel,
    pub spin: SpinSel,
}

impl Default for ChannelSet {
    fn default() -> Self {
        ChannelSet {
            irreps: IrrepSel::All,
            spin: SpinSel::RefOnly,
        }
    }
}

/// Per-channel ground-state options (the arm-3 recipe generalized: typed start → MALS ramp →
/// Riemannian-CG polish).
#[derive(Clone, Debug)]
pub struct ChannelGsOpts {
    /// Rank cap. Abelian: pooled bond dimension. SU(2): REDUCED MULTIPLET cap — the two are not
    /// comparable, and an SU(2) cap chosen by analogy with an abelian one is a trap.
    pub chi: Pooled,
    /// MALS rank-discovery `eps`.
    pub eps: f64,
    /// Explicit bond ramp; `None` derives one (abelian: doubling capped at `chi`; SU(2): the
    /// OVERSHOOT schedule, since the never-exceeding ramp is measured to trap at H2O scale).
    pub bond_ramp: Option<Vec<usize>>,
    /// MALS noise schedule.
    pub noise: Vec<f64>,
    /// Noise kind. `Perturbative` by default: `Random` has a seed-dependent trap in `|G| > 1`
    /// spaces, which is exactly this driver's regime.
    pub noise_type: NoiseType,
    pub max_iter: usize,
    /// Start-vector seed for channels that have no HF/CSF member.
    pub seed: u64,
    /// Riemannian-CG polish gradient tolerance.
    pub polish_gradtol: f64,
    /// Riemannian-CG restart budget.
    pub polish_restarts: usize,
    /// Above this `‖P_T grad‖` the channel is reported [`ChannelStatus::GsNotCritical`] — the
    /// cone readout still runs, with the caveat carried.
    pub grad_accept: f64,
}

impl Default for ChannelGsOpts {
    fn default() -> Self {
        ChannelGsOpts {
            chi: Pooled::NONE,
            eps: 1e-10,
            bond_ramp: None,
            noise: vec![1e-2, 1e-3, 1e-4, 0.0],
            noise_type: NoiseType::Perturbative,
            max_iter: 30,
            seed: 0xC4A1,
            polish_gradtol: 1e-13,
            polish_restarts: 30,
            grad_accept: 1e-9,
        }
    }
}

/// Options for [`channel_response`].
#[derive(Clone, Debug)]
pub struct ChannelOpts {
    pub channels: ChannelSet,
    /// Irrep NAMES only (labels / table printing); no reported number ever depends on it.
    pub kind: Option<PointGroupKind>,
    pub gs: ChannelGsOpts,
    /// The embedded per-channel cone solve. `max_added == 0` gives the tangent-only per-channel
    /// CI, which is still a valid composition mode.
    pub cone: ConeOpts,
    /// Hamiltonian assembly screen (the `eps` of `SparseHamiltonian::new`). This is the
    /// operator's own coefficient threshold and has nothing to do with the point-group
    /// screening contract above — `f64::EPSILON` does NOT remove a symmetry-forbidden 1e-14
    /// residue, which is exactly why `(t, v)` must arrive already screened.
    pub eps_h: f64,
}

impl Default for ChannelOpts {
    fn default() -> Self {
        ChannelOpts {
            channels: ChannelSet::default(),
            kind: None,
            gs: ChannelGsOpts::default(),
            cone: ConeOpts::default(),
            eps_h: f64::EPSILON,
        }
    }
}

/// A channel's identity: its point-group irrep and its spin offset from the reference sector.
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct ChannelLabel {
    pub irrep: Irrep,
    /// Rendered name from [`ChannelOpts::kind`] (e.g. `"B1"`); `None` when no kind was supplied.
    pub name: Option<String>,
    /// `0` for the reference spin sector.
    pub spin_delta: usize,
}

/// What happened in a channel.
#[derive(Clone, Debug, PartialEq)]
pub enum ChannelStatus {
    Ok,
    /// `max_ranks`-empty at bond `d`, or infeasible parameters (`ndn < delta`, `nup + delta > d`,
    /// `two_s + 2·delta` over the SU(2) cap, parity). Reported WITHOUT building an `H` and
    /// without letting a space constructor panic.
    Empty,
    /// The ground-state solve returned but `‖P_T grad‖ > gs.grad_accept`; the spectrum is
    /// reported with this caveat rather than dropped.
    GsNotCritical {
        grad_norm: f64,
    },
    /// The cone stage refused (ι-isometry / gauge-degenerate frames) or otherwise failed. No
    /// panic escapes the driver.
    Failed {
        message: String,
    },
}

/// One channel's result. Generic over the space descriptor so one definition serves both layers.
#[derive(Clone, Debug)]
pub struct ChannelResult<S> {
    pub label: ChannelLabel,
    /// The channel's typed space — `None` exactly when the channel's PARAMETERS were infeasible
    /// (`ndn < delta`, `nup + delta > d`, `two_s + 2·delta` over the SU(2) cap), so no space was
    /// ever constructed. `Some` otherwise, including for a [`ChannelStatus::Empty`] channel whose
    /// typed space exists but is `max_ranks`-empty at bond `d`.
    pub space: Option<S>,
    pub status: ChannelStatus,
    /// The channel's fixed-rank ground energy (no `e_core` — the caller adds it).
    pub e_gs: Option<f64>,
    pub grad_norm: Option<f64>,
    /// Present for [`ChannelStatus::Ok`] and [`ChannelStatus::GsNotCritical`].
    pub cone: Option<ConeResponse>,
}

/// One level of the merged spectrum.
#[derive(Clone, Copy, Debug)]
pub struct Level {
    /// Absolute energy: the channel's `E_0^CI`, or `E_0^CI + gaps[j]`.
    pub energy: f64,
    /// `energy − E_ref0`; `None` when the reference channel failed.
    pub gap: Option<f64>,
    /// Index into [`ChannelSpectrum::channels`].
    pub channel: usize,
    /// `0` = the channel's ground level, `j` = its `j`-th cone gap.
    pub k: usize,
    /// This level is a DEGENERATE COPY of a physical state that another enumerated channel also
    /// contains, so it must be dropped before the levels are read as a multiset. See
    /// [`ChannelSpectrum::levels`] for the mechanism; `false` on every irrep channel and on every
    /// SU(2) channel.
    pub redundant: bool,
}

/// The merged, labeled spectrum.
#[derive(Clone, Debug)]
pub struct ChannelSpectrum<S> {
    /// Channels in enumeration order — the reference channel first.
    pub channels: Vec<ChannelResult<S>>,
    /// Index of the reference channel in [`ChannelSpectrum::channels`].
    pub reference: usize,
    /// Every channel's levels merged and sorted ascending by absolute energy, ties broken by
    /// `(channel, k)` for determinism.
    pub levels: Vec<Level>,
}

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub(crate) struct ChannelPlan {
    pub(crate) spin_delta: usize,
    pub(crate) irrep: Irrep,
}

pub(crate) fn channel_plan(pg_order: usize, target: Irrep, opts: &ChannelOpts) -> Vec<ChannelPlan> {
    let irreps: Vec<Irrep> = match &opts.channels.irreps {
        IrrepSel::RefOnly => vec![target],
        IrrepSel::All => (0..pg_order).map(|i| Irrep::new(i as u8)).collect(),
        IrrepSel::List(list) => {
            let mut out: Vec<Irrep> = Vec::new();
            for &ir in list {
                if (ir.index() as usize) < pg_order && !out.contains(&ir) {
                    out.push(ir);
                }
            }
            out
        }
    };
    let dmax = match opts.channels.spin {
        SpinSel::RefOnly => 0,
        SpinSel::Raised(m) => m,
    };
    let mut plan = vec![ChannelPlan {
        spin_delta: 0,
        irrep: target,
    }];
    for delta in 0..=dmax {
        for &irrep in &irreps {
            let entry = ChannelPlan {
                spin_delta: delta,
                irrep,
            };
            if entry != plan[0] {
                plan.push(entry);
            }
        }
    }
    plan
}

impl ChannelPlan {
    pub(crate) fn label(&self, kind: Option<PointGroupKind>) -> ChannelLabel {
        ChannelLabel {
            irrep: self.irrep,
            name: kind
                .and_then(|k| k.irrep_name(self.irrep).ok())
                .map(|s| s.to_string()),
            spin_delta: self.spin_delta,
        }
    }
}

pub(crate) fn merge_levels<S>(
    channels: &[ChannelResult<S>],
    reference: usize,
    spin_channels_redundant: bool,
) -> Vec<Level> {
    let e_ref0 = channels
        .get(reference)
        .and_then(|c| c.cone.as_ref())
        .map(|r| r.e0);
    let mut levels: Vec<Level> = Vec::new();
    for (ci, c) in channels.iter().enumerate() {
        let Some(cone) = c.cone.as_ref() else {
            continue;
        };
        let redundant = spin_channels_redundant && c.label.spin_delta > 0;
        for (k, &energy) in cone.energies.iter().enumerate() {
            levels.push(Level {
                energy,
                gap: e_ref0.map(|r| energy - r),
                channel: ci,
                k,
                redundant,
            });
        }
    }
    levels.sort_by(|a, b| {
        a.energy
            .partial_cmp(&b.energy)
            .unwrap_or(std::cmp::Ordering::Equal)
            .then(a.channel.cmp(&b.channel))
            .then(a.k.cmp(&b.k))
    });
    levels
}

pub(crate) fn derived_ramp(chi: usize) -> Vec<usize> {
    if chi == 0 {
        return vec![4, 8, 16, 32, 0, 0];
    }
    (2..12).map(|k| chi.min(1usize << k)).collect()
}

pub(crate) fn start_rank(rmax: PerSector) -> usize {
    if rmax.is_none() { 8 } else { rmax.0.min(8) }
}

pub(crate) const START_PERTURBATION: f64 = 0.2;

fn determinant_irrep(space: &QnSpace, up: &[usize], dn: &[usize]) -> Irrep {
    up.iter()
        .chain(dn.iter())
        .fold(Irrep::IDENTITY, |ir, &s| ir.fuse(space.pg.irrep_of(s)))
}

fn determinant_start(space: QnSpace) -> Option<TtVector<f64>> {
    let d = space.d;
    let base_up: Vec<usize> = (0..space.nup).collect();
    let base_dn: Vec<usize> = (0..space.ndn).collect();
    let mut cands: Vec<(Vec<usize>, Vec<usize>)> = vec![(base_up.clone(), base_dn.clone())];
    for i in 0..space.nup {
        for a in space.nup..d {
            let mut up = base_up.clone();
            up[i] = a;
            cands.push((up, base_dn.clone()));
        }
    }
    for i in 0..space.ndn {
        for a in space.ndn..d {
            let mut dn = base_dn.clone();
            dn[i] = a;
            cands.push((base_up.clone(), dn));
        }
    }
    let (up, dn) = cands
        .into_iter()
        .find(|(up, dn)| determinant_irrep(&space, up, dn) == space.target)?;
    let mut occ_up = vec![false; d];
    let mut occ_dn = vec![false; d];
    for &s in &up {
        occ_up[s] = true;
    }
    for &s in &dn {
        occ_dn[s] = true;
    }
    Some(tt_state_in(space, &occ_up, &occ_dn))
}

fn channel_start(space: QnSpace, gs: &ChannelGsOpts) -> TtVector<f64> {
    let d = space.d;
    let caps = crate::qn::max_ranks(&space);
    let interior = start_rank(PerSector(1));
    let mut rng = Xoshiro256PlusPlus::seed_from_u64(gs.seed);
    let ranks: Vec<RankGrid> = (0..=d)
        .map(|b| {
            RankGrid::from_fn(&space, |s| {
                if b == 0 || b == d {
                    caps[b][s].min(1)
                } else {
                    caps[b][s].min(interior)
                }
            })
        })
        .collect();
    let Some(det) = determinant_start(space) else {
        return tt_randn(space, &ranks, &mut rng);
    };
    let mut x = crate::vector::perturbation(&det, &ranks, START_PERTURBATION, &mut rng);
    x.round_ranks_mut(&ranks);
    normalize_mut(&mut x);
    x
}

fn channel_space(base: QnSpace, entry: &ChannelPlan) -> Option<QnSpace> {
    let d = base.d;
    let delta = entry.spin_delta;
    if delta > base.ndn || base.nup + delta > d {
        return None;
    }
    Some(QnSpace::with_point_group(
        d,
        base.nup + delta,
        base.ndn - delta,
        base.pg,
        entry.irrep,
    ))
}

macro_rules! impl_channel {
    (
        ham: $ham:ty,
        vec: $vec:ty,
        space: $space:ty,
        ropts: $ropts:ident,
        channel_start: $channel_start:path,
        derived_ramp: $derived_ramp:path,
        channel_space: $channel_space:path,
        max_ranks: $max_ranks:path,
        mals: $mals:path,
        riemannian_cg: $riemannian_cg:path,
        exact_gradnorm: $exact_gradnorm:path,
        normalize_mut: $normalize_mut:path,
        rayleigh_quotient: $rayleigh_quotient:path,
        merge_degenerate: $merge_degenerate:expr,
        $(#[$doc_gs:meta])*
        channel_ground_state,
        $(#[$doc_cr:meta])*
        channel_response $(,)?
    ) => {
        $(#[$doc_gs])*
        pub fn channel_ground_state(
            h: &$ham,
            space: $space,
            gs: &ChannelGsOpts,
        ) -> ($vec, f64, f64) {
            let x0 = $channel_start(space, gs);
            let mopts = MalsOptions {
                eps: gs.eps,
                bond_dims: gs
                    .bond_ramp
                    .clone()
                    .unwrap_or_else(|| $derived_ramp(gs.chi.0)),
                noise: gs.noise.clone(),
                noise_type: gs.noise_type,
                seed: gs.seed,
                max_iter: gs.max_iter,
                ..MalsOptions::default()
            };
            let (_lam, xa) = $mals(h, &x0, &mopts);

            let mut ropts = $ropts::new(&space);
            ropts.eps = 1e-13;
            ropts.max_iter = 600;
            ropts.gradtol = gs.polish_gradtol;
            let stop = 0.1 * gs.grad_accept;
            let grad = |y: &$vec| $exact_gradnorm(h, y);
            let polish = |y0: &$vec| -> $vec {
                let (_l, mut y) = $riemannian_cg(h, y0, &ropts);
                for _ in 0..gs.polish_restarts {
                    if grad(&y) < stop {
                        break;
                    }
                    let (_l2, y2) = $riemannian_cg(h, &y, &ropts);
                    y = y2;
                }
                y
            };
            let mut x = polish(&xa);
            $normalize_mut(&mut x);
            x.round_mut(1e-8, Pooled::NONE, RoundMode::Rel);
            $normalize_mut(&mut x);
            let mut x = polish(&x);
            $normalize_mut(&mut x);
            x.check();
            let e = $rayleigh_quotient(h, &x);
            let g = grad(&x);
            (x, e, g)
        }

        $(#[$doc_cr])*
        pub fn channel_response(
            t: &CoeffTensor<f64>,
            v: &CoeffTensor<f64>,
            base: $space,
            opts: &ChannelOpts,
        ) -> ChannelSpectrum<$space> {
            let d = base.d;
            let plan = channel_plan(base.pg.order() as usize, base.target, opts);
            let mut channels: Vec<ChannelResult<$space>> = Vec::with_capacity(plan.len());

            for entry in plan {
                let label = entry.label(opts.kind);
                let Some(space) = $channel_space(base, &entry) else {
                    channels.push(ChannelResult {
                        label,
                        space: None,
                        status: ChannelStatus::Empty,
                        e_gs: None,
                        grad_norm: None,
                        cone: None,
                    });
                    continue;
                };

                let caps = $max_ranks(&space);
                if space.bond_sectors(d).iter().all(|&s| caps[d][s] == 0) {
                    channels.push(ChannelResult {
                        label,
                        space: Some(space),
                        status: ChannelStatus::Empty,
                        e_gs: None,
                        grad_norm: None,
                        cone: None,
                    });
                    continue;
                }

                let h = <$ham>::new(space, t, v, None, opts.eps_h);
                let (x, e_gs, grad_norm) = channel_ground_state(&h, space, &opts.gs);
                let (status, cone) = match try_cone_response(&h, &x, &opts.cone) {
                    Ok(r) => (
                        if grad_norm > opts.gs.grad_accept {
                            ChannelStatus::GsNotCritical { grad_norm }
                        } else {
                            ChannelStatus::Ok
                        },
                        Some(r),
                    ),
                    Err(refusal) => (
                        ChannelStatus::Failed {
                            message: refusal.message,
                        },
                        None,
                    ),
                };
                channels.push(ChannelResult {
                    label,
                    space: Some(space),
                    status,
                    e_gs: Some(e_gs),
                    grad_norm: Some(grad_norm),
                    cone,
                });
            }

            let levels = merge_levels(&channels, 0, $merge_degenerate);
            ChannelSpectrum {
                channels,
                reference: 0,
                levels,
            }
        }
    };
}

pub(crate) use impl_channel;

impl_channel!(
    ham: SparseHamiltonian<f64>,
    vec: TtVector<f64>,
    space: QnSpace,
    ropts: RiemannianOpts,
    channel_start: channel_start,
    derived_ramp: derived_ramp,
    channel_space: channel_space,
    max_ranks: crate::qn::max_ranks,
    mals: mals,
    riemannian_cg: riemannian_cg,
    exact_gradnorm: exact_gradnorm,
    normalize_mut: normalize_mut,
    rayleigh_quotient: rayleigh_quotient,
    merge_degenerate: true,
    channel_ground_state,
    channel_response,
);

/// [`channel_response`] from a loaded integrals bundle — SCREENED HERE, unconditionally.
///
/// # Panics
///
/// If the load's integrals contain a symmetry-forbidden entry above
/// [`crate::fcidump::SYMMETRY_FORBIDDEN_TOL`] — that is not roundoff residue but a genuine
/// mismatch between the coefficients and the resolved group, which the screen REFUSES rather than
/// silently zeroing. The panic names the offending entry; the alternative is the same run dying
/// later inside the automaton with no such information.
pub fn channel_response_from_load(
    load: &IntegralsLoad<f64>,
    opts: &ChannelOpts,
) -> ChannelSpectrum<QnSpace> {
    let mut opts = opts.clone();
    if opts.kind.is_none() {
        opts.kind = Some(load.pg.kind);
    }
    if load.space.pg.is_trivial() {
        return channel_response(&load.ints.t, &load.ints.v, load.space, &opts);
    }
    let mut t = load.ints.t.clone();
    let mut v = load.ints.v.clone();
    crate::fcidump::screen_point_group_forbidden(&mut t, &mut v, load.space.pg).unwrap_or_else(
        |e| {
            panic!(
                "channel_response_from_load: the loaded integrals are not compatible with the \
                 resolved point group ({e}). A ~1e-14 forbidden residue is screened silently; \
                 this entry is above the refusal tolerance, so the coefficients and the group \
                 disagree — check the dump's ORBSYM or the PointGroupPolicy."
            )
        },
    );
    channel_response(&t, &v, load.space, &opts)
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::solvers::hess::{ConeGuards, ConeResponse};

    fn cone_at(energies: &[f64]) -> ConeResponse {
        let e0 = energies[0];
        ConeResponse {
            gaps: energies[1..].iter().map(|e| e - e0).collect(),
            e0,
            energies: energies.to_vec(),
            seeded: false,
            added: 0,
            rounds: 0,
            guards: ConeGuards {
                iso: 0.0,
                asym: 0.0,
                ortho: 0.0,
                gram: 0.0,
            },
            williamson: None,
            character: Vec::new(),
        }
    }

    fn result(irrep: u8, spin_delta: usize, cone: Option<ConeResponse>) -> ChannelResult<()> {
        ChannelResult {
            label: ChannelLabel {
                irrep: Irrep::new(irrep),
                name: None,
                spin_delta,
            },
            space: Some(()),
            status: if cone.is_some() {
                ChannelStatus::Ok
            } else {
                ChannelStatus::Failed {
                    message: "synthetic".into(),
                }
            },
            e_gs: None,
            grad_norm: None,
            cone,
        }
    }

    #[test]
    fn merge_levels_orders_and_gaps_against_the_reference() {
        let channels = vec![
            result(0, 0, Some(cone_at(&[-2.0, -1.0]))),
            result(1, 0, Some(cone_at(&[-1.5, -1.0]))),
        ];
        let levels = merge_levels(&channels, 0, true);
        assert_eq!(
            levels
                .iter()
                .map(|l| (l.channel, l.k, l.redundant))
                .collect::<Vec<_>>(),
            vec![(0, 0, false), (1, 0, false), (0, 1, false), (1, 1, false)],
            "levels must sort by energy with (channel, k) as the deterministic tie-break"
        );
        for l in &levels {
            assert_eq!(
                l.gap,
                Some(l.energy + 2.0),
                "gaps are measured against e_ref0"
            );
        }
    }

    #[test]
    fn merge_levels_survives_a_failed_reference() {
        let channels = vec![
            result(0, 0, None),
            result(1, 0, Some(cone_at(&[-1.5, -1.0]))),
        ];
        let levels = merge_levels(&channels, 0, true);
        assert_eq!(
            levels.len(),
            2,
            "the failed reference contributes nothing; the live channel still reports"
        );
        assert!(
            levels.iter().all(|l| l.gap.is_none() && l.channel == 1),
            "with no reference e0 there is nothing to measure a gap against, so every gap is None"
        );
    }

    #[test]
    fn merge_levels_marks_only_abelian_spin_channels_redundant() {
        let channels = vec![
            result(0, 0, Some(cone_at(&[-2.0, -1.0]))),
            result(0, 1, Some(cone_at(&[-1.0, -0.5]))),
        ];
        let abelian = merge_levels(&channels, 0, true);
        assert_eq!(
            abelian
                .iter()
                .filter(|l| l.redundant)
                .map(|l| l.channel)
                .collect::<Vec<_>>(),
            vec![1, 1],
            "on a spin-free abelian H every level of the raised sector is a copy"
        );
        let su2 = merge_levels(&channels, 0, false);
        assert!(
            su2.iter().all(|l| !l.redundant),
            "SU(2) 2S channels partition the multiplet content — nothing is a copy there"
        );
    }

    #[test]
    fn channel_plan_list_filters_out_of_range_and_dedups() {
        let opts = ChannelOpts {
            channels: ChannelSet {
                irreps: IrrepSel::List(vec![
                    Irrep::new(3),
                    Irrep::new(1),
                    Irrep::new(3),
                    Irrep::new(9),
                ]),
                spin: SpinSel::RefOnly,
            },
            ..Default::default()
        };
        let plan = channel_plan(4, Irrep::new(1), &opts);
        assert_eq!(
            plan.iter()
                .map(|p| (p.irrep.index(), p.spin_delta))
                .collect::<Vec<_>>(),
            vec![(1, 0), (3, 0)],
            "the reference comes first, irrep 9 is out of range for order 4, and the second \
             irrep-3 entry is a duplicate — the caller's order otherwise stands"
        );
    }

    #[test]
    fn channel_plan_enumerates_the_reference_first() {
        let opts = ChannelOpts {
            channels: ChannelSet {
                irreps: IrrepSel::All,
                spin: SpinSel::Raised(1),
            },
            ..Default::default()
        };
        let plan = channel_plan(2, Irrep::new(1), &opts);
        assert_eq!(
            plan.iter()
                .map(|p| (p.spin_delta, p.irrep.index()))
                .collect::<Vec<_>>(),
            vec![(0, 1), (0, 0), (1, 0), (1, 1)],
            "the reference (0, target) must be enumerated first and exactly once"
        );
    }
}
