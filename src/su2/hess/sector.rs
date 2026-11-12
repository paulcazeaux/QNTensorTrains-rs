// Copyright (C) 2023-2026 Paul Cazeaux
//
// This file is part of qntt. qntt is free software: you can
// redistribute it and/or modify it under the terms of the GNU
// General Public License as published by the Free Software
// Foundation, either version 3 of the License, or (at your
// option) any later version. See LICENSE for details.

use rand::SeedableRng;
use rand_xoshiro::Xoshiro256PlusPlus;

use crate::solvers::davidson_block::{BlockDavidsonOpts, davidson_block_projected};
use crate::solvers::hess::{
    ModeCheck, ModeCheckOps, ModeCheckOpts, NegativeWitness, RpaAmplitudes, RpaMetric, RpaMode,
    RpaModeKind, RpaReadout, SectorOpts, SectorPencil, SectorReadout, SectorSpectrum,
    SymplecticSpectrum, TanOp, certify_negatives, check_sector_kernel, mode_checks_core,
    resolve_sector_precond, sector_d0_precond, sector_rpa_solve, symplectic_from_grams,
};
use crate::su2::core::Su2Core;
use crate::su2::flux::Su2ShiftedFrames;
use crate::su2::hamiltonian::Su2SparseHamiltonian;
use crate::su2::hamiltonian::matvec::{Su2OneSiteWorkspace, su2_onesite_workspace};
use crate::su2::qn::{Su2RankGrid, Su2Space};
use crate::su2::solvers::als::{left_to_right_sketching, right_to_left_sketching};
use crate::su2::tangent::Su2TtTangent;
use crate::su2::vector::{Su2TtVector, tt_randn};
use crate::su2::weingarten::{
    Su2FusedTermA, min_schmidt, tangent_basis_shifted, tangent_dim_grids,
};

use super::{Su2ShiftedEnvs, Su2WeingartenCtx, effham_apply_shifted_dressed, effham_gram_shifted};

/// The outcome of one SPIN channel's readout, with the conditioning of the base point reported
/// beside the numbers (a spectral value at an ill-conditioned frame is not quotable on its
/// own).
#[derive(Clone, Debug)]
pub struct Su2SectorResponse {
    /// The numbers, tagged by readout — read them through [`Self::theta`] / [`Self::omega2`] /
    /// [`Self::omega`] / [`Self::rpa`], which cannot hand back a value in the wrong units.
    pub spectrum: SectorSpectrum,
    /// `dim T^{S'}` ([`tangent_dim_grids`]) — the exact-exhaustion cap of the channel, STRUCTURAL
    /// (shapes only). NOT the target sector's dimension: compare it against the MULTIPLET count of
    /// the `2S = 2κ` sector before reading the spectrum as that sector's levels (module header).
    pub dim: usize,
    /// The channel's total doubled spin `2S' = 2κ` — which is also the flux, because the reference
    /// is a singlet.
    pub two_s_out: usize,
    /// The space the response reaches, `x.space.with_two_s(2κ)`: same `N`, same `d`, same point
    /// group and same target irrep, raised total spin.
    pub space: Su2Space,
    /// Every requested root came back converged (on the `Rpa` arm, the readout's `converged`).
    pub converged: bool,
    /// Outer (Rayleigh–Ritz) iterations (`Tda`) / basis vectors built (`Rpa`, either kernel).
    pub iterations: usize,
    /// Applies consumed, INCLUDING the fresh ones the negative certificates spent; on the `Rpa`
    /// arm in `(A, B)` pairs, as the abelian [`crate::solvers::hess::SectorResponse::applies`].
    pub applies: usize,
    /// Smallest CENTERWEIGHTED bond Schmidt value of the BASE point
    /// ([`crate::su2::weingarten::min_schmidt`]) — base-point data, identical across channels,
    /// quoted with every reading. Centerweighted, never a raw `σ`: on this layer a raw block at
    /// column spin `2s` stands for a `(2s+1)`-fold multiplet, so a raw fold would understate the
    /// conditioning risk by up to `√(2S+1)` in the low-`2s` sectors.
    pub sigma_min: f64,
}

impl Su2SectorResponse {
    /// The `Tda` arm's `eig(A^{S'})`, or `None` on an `Rpa` readout — `ω²` is not an energy.
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

    /// The `Rpa` arm's signed `ω²`, or `None` on a `Tda` readout. [`Su2SectorCtx::sector_lowest`]
    /// produces it and [`Su2SectorCtx::sector_lowest_tda`] does not — one readout per entry point,
    /// each refusing the other's — so which arm is live is decided by the CALL, never by the
    /// channel. An empty `Some(vec![])` is a real answer in `Rpa` units (two routes, see that
    /// method's doc); `None` means the readout was `Tda`.
    pub fn omega2(&self) -> Option<Vec<f64>> {
        self.rpa().map(RpaReadout::omega2)
    }

    /// `ω` — the readout's publication gate ([`RpaReadout::omega`]), the abelian
    /// [`crate::solvers::hess::SectorResponse::omega`]'s rule verbatim: a de-excitation, an
    /// imaginary or complex mode, a metric event or a counting offset withholds it.
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

    /// Every negative value carries a certified fresh-apply witness (vacuously true when there are
    /// none) — the publication gate, on either arm.
    pub fn all_negatives_certified(&self) -> bool {
        match &self.spectrum {
            SectorSpectrum::Tda { negative, .. } => negative.iter().all(NegativeWitness::certified),
            SectorSpectrum::Rpa(r) => r.all_negatives_certified(),
        }
    }

    /// The SIGNED solve met an indefinite metric (a metric event in its report); always `false`
    /// on a paired readout, which has no metric — read [`Self::anomalous`].
    pub fn metric_indefinite(&self) -> bool {
        self.rpa().is_some_and(|r| r.metric_events() > 0)
    }

    /// The `Rpa` solve found anything a stable channel would not show
    /// ([`RpaReadout::anomalous`]) — the kernel-independent flag, the abelian
    /// [`crate::solvers::hess::SectorResponse::anomalous`] verbatim. `false` on a `Tda` readout.
    pub fn anomalous(&self) -> bool {
        self.rpa().is_some_and(RpaReadout::anomalous)
    }

    /// The projected pencil's inertia, a lower bound on `n₋(A^{S'} + B^{S'}) + n₋(A^{S'} − B^{S'})`
    /// — the abelian [`crate::solvers::hess::SectorResponse::index_lower_bound`]'s counting
    /// verbatim. `None` on a `Tda` readout.
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

/// One SPIN response channel bound at a fixed singlet reference: the frames and `λ` of a
/// [`Su2WeingartenCtx`], the flux `2κ`, and the Hamiltonian rebuilt on
/// `space_out = space.with_two_s(2κ)`.
pub struct Su2SectorCtx<'c, 'h, T: crate::scalar::Scalar> {
    ctx: &'c Su2WeingartenCtx<'h, T>,
    h_out: &'h Su2SparseHamiltonian<T>,
    two_s_flux: usize,
    dressed: Su2ShiftedFrames<T>,
    envs: std::sync::OnceLock<Su2ShiftedEnvs<T>>,
}

impl SectorPencil for Su2SectorCtx<'_, '_, f64> {
    type V = Su2TtTangent<f64>;
    fn a_apply(&self, v: &Su2TtTangent<f64>) -> Su2TtTangent<f64> {
        Su2SectorCtx::a_apply(self, v)
    }
    fn b_apply(&self, v: &Su2TtTangent<f64>) -> Su2TtTangent<f64> {
        Su2SectorCtx::b_apply(self, v)
    }
    fn hpp_apply(&self, v: &Su2TtTangent<f64>) -> Su2TtTangent<f64> {
        Su2SectorCtx::hpp_apply(self, v)
    }
    fn hmm_apply(&self, v: &Su2TtTangent<f64>) -> Su2TtTangent<f64> {
        Su2SectorCtx::hmm_apply(self, v)
    }
    fn a_diagonal(&self) -> Su2TtTangent<f64> {
        Su2SectorCtx::a_diagonal(self)
    }
    fn a_diagonal_squared(&self) -> Su2TtTangent<f64> {
        Su2SectorCtx::a_diagonal_squared(self)
    }
    fn clean(v: &mut Su2TtTangent<f64>) {
        v.gauge_fix_orthogonal_mut();
    }
}

impl<'c, 'h> Su2SectorCtx<'c, 'h, f64> {
    /// Bind the `2κ` spin channel at the reference `ctx` was built on.
    pub fn new(
        ctx: &'c Su2WeingartenCtx<'h, f64>,
        h_out: &'h Su2SparseHamiltonian<f64>,
        two_s_flux: usize,
    ) -> Self {
        assert_ne!(
            two_s_flux, 0,
            "Su2SectorCtx: 2S_flux = 0 is not a cross-sector channel — T^{{S=0}} is x's ordinary \
             tangent space and A^{{S=0}} its ordinary effective Hamiltonian. Use \
             su2::hess::weingarten_operator(h, x) and the shipped in-sector RPA/TDA readouts"
        );
        let (base_l, _base_r) = ctx.bases();
        let space = base_l.space;
        assert_eq!(
            space.two_s, 0,
            "Su2SectorCtx: the reference must be a singlet (2S = 0), got 2S = {}. The shifted \
             derivation identifies the bond label's spin with the borrowed suffix multiplet's spin, \
             which holds only at 2S = 0; for 2S > 0 the suffix frame is an INTERTWINER and the \
             borrowing needs a second, genuinely three-index recoupling that does not exist yet",
            space.two_s
        );
        super::assert_shifted_operator(h_out, base_l, two_s_flux);
        let (_, base_r) = ctx.bases();
        let dressed = Su2ShiftedFrames::new(base_l, base_r, two_s_flux);
        Su2SectorCtx {
            ctx,
            h_out,
            two_s_flux,
            dressed,
            envs: std::sync::OnceLock::new(),
        }
    }

    /// The channel's total doubled spin `2S' = 2κ` (equal to the flux, the reference being a
    /// singlet).
    #[inline]
    pub fn two_s_out(&self) -> usize {
        self.two_s_flux
    }

    /// The Rayleigh quotient at the reference — the shift in `A^{S'} = P^{S'}(H_out − λ)P^{S'}`.
    #[inline]
    pub fn lambda(&self) -> f64 {
        self.ctx.lambda
    }

    /// `space_out = x.space.with_two_s(2κ)`: the space `ι^{S'}(ξ)` and `h_out` live in.
    #[inline]
    pub fn space_out(&self) -> Su2Space {
        self.h_out.space
    }

    /// The ORIGINAL gauge frames of `x`, in `x`'s own space — not the shifted pair, which the
    /// crate-internal `su2::tangent::shifted_frames` builds wherever it is needed.
    #[inline]
    pub fn bases(&self) -> (&Su2TtVector<f64>, &Su2TtVector<f64>) {
        self.ctx.bases()
    }

    fn envs(&self) -> &Su2ShiftedEnvs<f64> {
        self.envs.get_or_init(|| {
            let h = self.h_out;
            let (u, v) = &self.dressed.frames;
            let out_space = self.dressed.out_space;
            let d = out_space.d;
            let rm = &self.dressed.rm;
            let rd = &self.dressed.rd;

            let ws_u: Vec<Su2OneSiteWorkspace> = (0..d)
                .map(|k| su2_onesite_workspace(h, u.core(k)))
                .collect();
            let ws_v: Vec<Su2OneSiteWorkspace> = (0..d)
                .map(|k| su2_onesite_workspace(h, v.core(k)))
                .collect();
            let ws_du: Vec<Su2OneSiteWorkspace> = (0..d)
                .map(|k| {
                    let decl = Su2Core::<f64>::zeros(out_space, k, &rm[k], &rd[k + 1]);
                    su2_onesite_workspace(h, &decl)
                })
                .collect();

            let l = left_to_right_sketching(h, u);
            let r_kappa = right_to_left_sketching(h, v);

            Su2ShiftedEnvs {
                l,
                r_kappa,
                ws_u,
                ws_v,
                ws_du,
            }
        })
    }

    /// `P^{S'}(y)` for `y ∈ space_out` — [`Su2TtTangent::project_shifted`], on the DRESSED pair this
    /// context already holds (so `V^{(1)}` is not rebuilt per call; `project_impl` is the body
    /// `project_shifted` reaches after dressing, and the values are bit-identical).
    pub fn project(&self, y: &Su2TtVector<f64>) -> Su2TtTangent<f64> {
        let (u, v) = &self.dressed.frames;
        Su2TtTangent::project_impl(u.clone(), v.clone(), self.two_s_flux, y)
    }

    /// `A^{S'}·ξ = P^{S'}((H_out − λ)·ι^{S'}(ξ))` — the FUSED environment sweep of
    /// [`Self::a_apply_fused`], with `QNTT_A_APPLY_EXACT` restoring the unfused reference product
    /// [`Self::a_apply_exact`].
    pub fn a_apply(&self, xi: &Su2TtTangent<f64>) -> Su2TtTangent<f64> {
        if crate::knobs::A_APPLY_EXACT.get() {
            return self.a_apply_exact(xi);
        }
        self.a_apply_fused(xi)
    }

    /// `A^{S'}·ξ` as a FUSED environment sweep — **`h_out·ι^{S'}(ξ)` is never formed** — bypassing
    /// the `QNTT_A_APPLY_EXACT` switch. The entry point the equivalence tests pin against
    /// [`Self::a_apply_exact`] / [`super::effham_apply_shifted`].
    pub fn a_apply_fused(&self, xi: &Su2TtTangent<f64>) -> Su2TtTangent<f64> {
        assert_eq!(
            xi.two_s_flux(),
            self.two_s_flux,
            "Su2SectorCtx::a_apply_fused: the direction carries spin flux {} but the channel is {} \
             — an A^S' applied across fluxes is not this channel's operator",
            xi.two_s_flux(),
            self.two_s_flux
        );
        let h = self.h_out;
        let e = self.envs();
        let (u, v) = &self.dressed.frames;

        let components = Su2FusedTermA {
            h,
            u,
            v,
            l: &e.l,
            r: &e.r_kappa,
            ws_u: &e.ws_u,
            ws_v: &e.ws_v,
            ws_du: &e.ws_du,
        }
        .components(&xi.components);

        let mut out = Su2TtTangent::from_parts_shifted(
            self.dressed.rm.clone(),
            self.dressed.rd.clone(),
            self.two_s_flux,
            u.clone(),
            v.clone(),
            components,
        );
        out.axpy(-self.lambda(), xi);
        out
    }

    /// `D_A^{S'}` — the EXACT coefficient-basis diagonal of `A^{S'}`'s local operator, the spin
    /// channel's Jacobi model behind `SympPrecond::DiagUnshifted` on both readouts
    /// (`D0^{S'}`, rpa-floor-selection P4; the SU(2) twin of the abelian `SectorCtx::a_diagonal`).
    pub fn a_diagonal(&self) -> Su2TtTangent<f64> {
        let h = self.h_out;
        let e = self.envs();
        let (u, v) = &self.dressed.frames;
        let out_space = self.dressed.out_space;
        let d = out_space.d;
        let rm = &self.dressed.rm;
        let rd = &self.dressed.rd;
        let components: Vec<Su2Core<f64>> = (0..d)
            .map(|k| {
                let decl = Su2Core::<f64>::zeros(out_space, k, &rm[k], &rd[k + 1]);
                h.effective_diagonal(&e.l[k], &e.r_kappa[k], &decl, &e.ws_du[k])
            })
            .collect();
        let mut diag = Su2TtTangent::from_parts_shifted(
            rm.clone(),
            rd.clone(),
            self.two_s_flux,
            u.clone(),
            v.clone(),
            components,
        );
        let shift = -self.lambda();
        diag.for_each_coeff_mut(|c| *c += shift);
        diag
    }

    fn a_diagonal_squared(&self) -> Su2TtTangent<f64> {
        let mut d = self.a_diagonal();
        d.for_each_coeff_mut(|c| {
            let sq = *c * *c;
            *c = sq;
        });
        d
    }

    /// `A^{S'}·ξ` by the UNFUSED reference product — [`super::effham_apply_shifted`] at these
    /// frames, `2κ` and `λ`, entered through its already-dressed body
    /// ([`effham_apply_shifted_dressed`]): embed at the summed bond rank, apply `h_out` exactly,
    /// project, subtract `λ·ξ`.
    pub fn a_apply_exact(&self, xi: &Su2TtTangent<f64>) -> Su2TtTangent<f64> {
        effham_apply_shifted_dressed(
            self.h_out,
            (&self.dressed.frames.0, &self.dressed.frames.1),
            self.two_s_flux,
            self.lambda(),
            xi,
        )
    }

    /// The row-major `n×n` Gram of `A^{S'}` over `basis` (CENTERWEIGHTED metric) — `n` FUSED
    /// applies plus `n²` tangent dots, the A-half of [`Self::grams`] exactly.
    pub fn a_gram(&self, basis: &[Su2TtTangent<f64>]) -> Vec<f64> {
        for (i, b) in basis.iter().enumerate() {
            assert_eq!(
                b.two_s_flux(),
                self.two_s_flux,
                "Su2SectorCtx::a_gram: basis vector {i} carries spin flux {} but the channel is \
                 {} — a Gram mixing fluxes is not the channel's operator",
                b.two_s_flux(),
                self.two_s_flux
            );
        }
        let n = basis.len();
        let pure: Vec<Su2TtTangent<f64>> =
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

    /// The row-major `n×n` Gram of `A^{S'}` by the UNFUSED reference route —
    /// [`effham_gram_shifted`] at these frames. The oracle [`Self::a_gram`] is pinned against.
    pub fn a_gram_exact(&self, basis: &[Su2TtTangent<f64>]) -> Vec<f64> {
        let (bl, br) = self.bases();
        effham_gram_shifted(self.h_out, bl, br, self.two_s_flux, self.lambda(), basis)
    }

    /// An orthonormal basis of `T^{S'}` — [`tangent_basis_shifted`]. `O(dim)` tangents live at once;
    /// the iterative readout below does not use it.
    pub fn basis(&self) -> Vec<Su2TtTangent<f64>> {
        let (bl, br) = self.bases();
        tangent_basis_shifted(bl, br, self.two_s_flux)
    }

    /// `dim T^{S'}` — [`tangent_dim_grids`], STRUCTURAL (shapes only). The MGS cull can leave
    /// [`Self::basis`] SHORTER at a truncated or specially-structured point.
    pub fn dim(&self) -> usize {
        let (bl, _) = self.bases();
        tangent_dim_grids(&bl.space, &bl.ranks, self.two_s_flux)
    }

    fn seed_block(&self, n: usize, seed: u64) -> Vec<Su2TtTangent<f64>> {
        let space_out = self.space_out();
        let d = space_out.d;
        let rows = &self.dressed.frames.0.ranks;
        let cols = &self.dressed.frames.1.ranks;
        let grid: Vec<Su2RankGrid> = (0..=d)
            .map(|b| {
                let mut g = Su2RankGrid::zeros(&space_out);
                for s in space_out.bond_sectors(b) {
                    g[s] = match b {
                        0 => rows[b][s],
                        _ if b == d => cols[b][s],
                        _ => rows[b][s] + cols[b][s],
                    };
                }
                g
            })
            .collect();
        let mut rng = Xoshiro256PlusPlus::seed_from_u64(seed);
        (0..n)
            .filter_map(|_| {
                let g = tt_randn::<f64, _>(space_out, &grid, &mut rng);
                let n0 = g.norm_lr(false);
                let v = self.project(&g);
                (v.norm() > f64::EPSILON.sqrt() * n0).then_some(v)
            })
            .collect()
    }

    /// The `k` lowest eigenvalues of `A^{S'}` — the channel's TDA-like excitation energies — by block
    /// Davidson restricted to `range(P^{S'})`.
    pub fn sector_lowest_tda(&self, k: usize, opts: &SectorOpts) -> Su2SectorResponse {
        assert!(
            opts.readout == SectorReadout::Tda,
            "sector_lowest_tda: the symplectic (RPA) readout of the pair (A^S', B^S') \
             is sector_lowest's; this entry point reports \
             eig(A^S') only, and returning an energy under an RPA label would mislabel the quantity"
        );
        let precond_choice =
            resolve_sector_precond(opts.precond, "Su2SectorCtx::sector_lowest_tda");
        let (base_l, base_r) = self.bases();
        let sigma_min = min_schmidt(base_l, base_r);
        let dim = self.dim();
        let empty = |converged: bool| Su2SectorResponse {
            spectrum: SectorSpectrum::Tda {
                theta: Vec::new(),
                negative: Vec::new(),
            },
            dim,
            two_s_out: self.two_s_flux,
            space: self.space_out(),
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
        let mu =
            crate::solvers::hess::complement_shift(crate::solvers::hess::diag_inf_scale(&dmodel));
        let mut d0 = sector_d0_precond(precond_choice, move || dmodel);
        let mut applies = 0usize;
        let mut op = |xi: &Su2TtTangent<f64>| {
            applies += 1;
            let mut q = xi.clone();
            q.gauge_fix_orthogonal_mut();
            let mut aq = self.a_apply(&q);
            crate::solvers::hess::add_complement_shift(&mut aq, xi, q, mu);
            aq
        };
        let mut precond = |t: &mut Su2TtTangent<f64>, theta: f64| {
            if let Some(p) = d0.as_mut() {
                p(t, theta);
            }
        };
        let mut project = |t: &mut Su2TtTangent<f64>| t.gauge_fix_orthogonal_mut();
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
            &mut |v: &Su2TtTangent<f64>| {
                applies += 1;
                self.a_apply(v)
            },
            &mut |v: &Su2TtTangent<f64>| v.clone(),
            &mut |_v: &mut Su2TtTangent<f64>| {},
            &mut |v: &mut Su2TtTangent<f64>| v.gauge_fix_orthogonal_mut(),
        );

        Su2SectorResponse {
            spectrum: SectorSpectrum::Tda { theta, negative },
            dim,
            two_s_out: self.two_s_flux,
            space: self.space_out(),
            converged: info.converged,
            iterations: info.niter,
            applies,
            sigma_min,
        }
    }

    /// `B^{S'}·ξ` — the spin-flux-`2κ` block of the Weingarten (shape-operator) map
    /// `B''ξ = P''(D_ξP'')r⊥` of the spin-UNTYPED manifold through `x`.
    pub fn b_apply(&self, xi: &Su2TtTangent<f64>) -> Su2TtTangent<f64> {
        assert_eq!(
            xi.two_s_flux(),
            self.two_s_flux,
            "Su2SectorCtx::b_apply: the direction carries spin flux {} but the channel is {} — \
             a B^S' applied across fluxes is not this channel's operator",
            xi.two_s_flux(),
            self.two_s_flux
        );
        self.ctx.b_apply_any(xi, Some(&self.dressed))
    }

    /// The whole Riemannian Hessian block `H^rr = A^{S'}·ξ + B^{S'}·ξ`.
    pub fn hess_apply(&self, xi: &Su2TtTangent<f64>) -> Su2TtTangent<f64> {
        self.a_apply(xi).add(&self.b_apply(xi))
    }

    /// `H^rr = A^{S'} + B^{S'}` — an alias of [`Self::hess_apply`], named for the symplectic
    /// driver's outer factor.
    pub fn hpp_apply(&self, xi: &Su2TtTangent<f64>) -> Su2TtTangent<f64> {
        self.hess_apply(xi)
    }

    /// `H^ii = A^{S'} − B^{S'}` — the symplectic driver's inner factor.
    pub fn hmm_apply(&self, xi: &Su2TtTangent<f64>) -> Su2TtTangent<f64> {
        self.a_apply(xi).sub(&self.b_apply(xi))
    }

    /// The row-major `n×n` Grams `A_ij = ⟨b_i, A^{S'} b_j⟩` and `B_ij = ⟨b_i, B^{S'} b_j⟩` over a
    /// `⟨·,·⟩`-orthonormal basis of `T^{S'}`, in the CENTERWEIGHTED tangent metric.
    pub fn grams(&self, basis: &[Su2TtTangent<f64>]) -> (Vec<f64>, Vec<f64>) {
        for (i, b) in basis.iter().enumerate() {
            assert_eq!(
                b.two_s_flux(),
                self.two_s_flux,
                "Su2SectorCtx::grams: basis vector {i} carries spin flux {} but the channel is {} \
                 — a Gram mixing fluxes is not the channel's operator",
                b.two_s_flux(),
                self.two_s_flux
            );
        }
        let n = basis.len();
        let pure: Vec<Su2TtTangent<f64>> =
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

    /// The DENSE symplectic (RPA) spectrum of the pair `(A^{S'}, B^{S'})` over `basis` — the
    /// certified oracle the iterative [`Self::sector_lowest`] must reproduce.
    pub fn symplectic_eigenvalues(&self, basis: &[Su2TtTangent<f64>]) -> SymplecticSpectrum {
        let (a, b) = self.grams(basis);
        symplectic_from_grams(&a, &b, basis.len(), 1e-10)
    }

    /// The `k` lowest symplectic (RPA) values of the channel — matrix-free, by the kernel
    /// [`SectorOpts::kernel`] names (the PAIRED-basis Davidson by default, the signed W-metric one
    /// through `RpaKernel::signed(lane)`) — reported as SIGNED `ω²`:
    /// [`Self::sector_lowest_vectors`] without the amplitudes. The abelian
    /// [`crate::solvers::hess::SectorCtx::sector_lowest`]'s contract on the spin channel, through
    /// the SAME solve (`sector_rpa_solve`): the kernels, the list, the Krein types, the flags and
    /// the two empty routes are documented there and not restated. There is no spin-flux refusal on
    /// this path (the SU(2) TT paired entry's refusal guards a reference-space seed, which a
    /// channel never takes).
    pub fn sector_lowest(&self, k: usize, opts: &SectorOpts) -> Su2SectorResponse {
        self.sector_lowest_vectors(k, opts).0
    }

    /// [`Self::sector_lowest`] that also returns each mode's **amplitudes** `(X, Y)`
    /// ([`RpaAmplitudes`]), one per entry in the same order, inside `T^{S'}`: the pair the paired
    /// kernel solved for, or the signed kernel's metric-side Ritz vector completed by its own
    /// W-image (`p = Wq/ω`). Zero extra applies. Their
    /// consumer is [`Self::rpa_mode_checks`]. On a regularized context every number and vector is
    /// the regularized pencil's ([`Self::sector_lowest`], "Regularized contexts"). Both empty routes
    /// return no mode.
    pub fn sector_lowest_vectors(
        &self,
        k: usize,
        opts: &SectorOpts,
    ) -> (Su2SectorResponse, Vec<RpaAmplitudes<Su2TtTangent<f64>>>) {
        assert!(
            opts.readout == SectorReadout::Rpa,
            "sector_lowest: the TDA readout eig(A^S') — an ENERGY — is sector_lowest_tda's; this \
             entry point reports the symplectic omega^2 of the pair (A^S', B^S'), and returning one \
             under the other's label would mislabel the quantity"
        );
        let precond_choice = resolve_sector_precond(opts.precond, "Su2SectorCtx::sector_lowest");
        check_sector_kernel(opts.kernel, "Su2SectorCtx::sector_lowest");
        let (base_l, base_r) = self.bases();
        let sigma_min = min_schmidt(base_l, base_r);
        let dim = self.dim();
        let empty = |converged: bool| Su2SectorResponse {
            spectrum: SectorSpectrum::Rpa(RpaReadout::empty(converged, opts.kernel)),
            dim,
            two_s_out: self.two_s_flux,
            space: self.space_out(),
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
        let resp = Su2SectorResponse {
            dim,
            two_s_out: self.two_s_flux,
            space: self.space_out(),
            converged: readout.converged,
            iterations: readout.iterations,
            applies: readout.applies,
            sigma_min,
            spectrum: SectorSpectrum::Rpa(readout),
        };
        (resp, modes)
    }

    /// The per-mode acceptance numbers of a signed SPIN-channel solve — the abelian
    /// [`crate::solvers::hess::SectorCtx::rpa_mode_checks`] on this layer (same body,
    /// `mode_checks_core`; the numbers are [`crate::solvers::hess::rpa_mode_checks`]'s): `W`/`R` and
    /// the `Y` sign by the solve's lane, the channel's `A^{S'}`/`B^{S'}`, `reproject = P^{S'}∘ι` and
    /// the IDENTITY deflate. An EMPTY `omega2` returns `[]` before the lane is read.
    ///
    /// # Panics
    ///
    /// On a `Tda` response and on `modes.len() != omega2.len()`.
    pub fn rpa_mode_checks(
        &self,
        resp: &Su2SectorResponse,
        modes: &[RpaAmplitudes<Su2TtTangent<f64>>],
        opts: &ModeCheckOpts,
    ) -> Vec<ModeCheck> {
        let _ = opts;
        let readout = match &resp.spectrum {
            SectorSpectrum::Rpa(r) => r,
            SectorSpectrum::Tda { .. } => panic!(
                "Su2SectorCtx::rpa_mode_checks: a Tda response — eig(A^S'), an ENERGY — has no \
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
        let hpp = |v: &Su2TtTangent<f64>| self.hpp_apply(v);
        let hmm = |v: &Su2TtTangent<f64>| self.hmm_apply(v);
        let (w, r, y_sign): (
            &TanOp<'_, Su2TtTangent<f64>>,
            &TanOp<'_, Su2TtTangent<f64>>,
            f64,
        ) = match lane {
            RpaMetric::Rr => (&hpp, &hmm, -1.0),
            RpaMetric::Ii => (&hmm, &hpp, 1.0),
            RpaMetric::Auto => panic!(
                "Su2SectorCtx::rpa_mode_checks: a non-empty Rpa readout whose lane was never \
                 resolved"
            ),
        };
        let a = |v: &Su2TtTangent<f64>| self.a_apply(v);
        let b = |v: &Su2TtTangent<f64>| self.b_apply(v);
        let reproject = |v: &mut Su2TtTangent<f64>| *v = self.project(&v.to_vector());
        let deflate = |_v: &mut Su2TtTangent<f64>| {};
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
mod csa_bundle_tests {
    use super::Su2SectorCtx;
    use crate::su2::hamiltonian::Su2SparseHamiltonian;
    use crate::su2::hess::weingarten_operator as su2_weingarten_operator;
    use crate::su2::qn::{Su2RankGrid, Su2Space, max_ranks as su2_max_ranks};
    use crate::su2::riemannian::tangent_bases as su2_tangent_bases;
    use crate::su2::vector::{Su2TtVector, tt_randn as su2_tt_randn};
    use crate::su2::weingarten::{tangent_basis_shifted, tangent_dim_grids};
    use crate::test_util::{generic_su2_h, randomized_su2_state, seeded};

    pub(super) fn live_channel_fixture() -> (Su2Space, Su2SparseHamiltonian<f64>, Su2TtVector<f64>)
    {
        let space = Su2Space::new(4, 4, 0);
        let ham = generic_su2_h::<f64>(space);
        let mut rng = seeded(0xC5_A4);
        let mut x: Su2TtVector<f64> = randomized_su2_state(space, 1, 1, &mut rng);
        x.scale_mut(1.0 / x.norm_lr(false));
        assert!(x.check(), "the fixture must be a legal singlet train");
        (space, ham, x)
    }

    pub(super) fn zero_width_channel_fixture()
    -> (Su2Space, Su2SparseHamiltonian<f64>, Su2TtVector<f64>) {
        let space = Su2Space::new(4, 2, 0);
        let ham = generic_su2_h::<f64>(space);
        let caps = su2_max_ranks(&space);
        let grids: Vec<Su2RankGrid> = (0..=space.d)
            .map(|b| {
                let mut r = Su2RankGrid::zeros(&space);
                for s in space.bond_sectors(b) {
                    if b == 0 || b == space.d || s.two_s == 0 {
                        r[s] = caps[b][s].min(1);
                    }
                }
                r
            })
            .collect();
        let mut rng = seeded(0xC5_A2);
        let mut x: Su2TtVector<f64> = su2_tt_randn(space, &grids, &mut rng);
        x.scale_mut(1.0 / x.norm_lr(false));
        assert!(
            x.check(),
            "the zero-width fixture must be a legal singlet train"
        );
        (space, ham, x)
    }

    fn table_width(tab: &crate::su2::qn::Su2SectorGrid<Vec<std::ops::Range<usize>>>) -> usize {
        tab.iter()
            .map(|(_, rs)| rs.iter().map(|q| q.end - q.start).sum::<usize>())
            .sum()
    }

    #[test]
    fn csa_env_bundle_shapes_and_degenerate_workspaces() {
        let (space, ham, x) = live_channel_fixture();
        let d = space.d;
        let two_k = 2usize;
        let (bl, _br) = su2_tangent_bases(&x);
        let dim = tangent_dim_grids(&space, &bl.ranks, two_k);
        assert!(
            dim > 0,
            "ANTI-VACUITY: the fixture's triplet channel must be LIVE (measured 13), or the \
             bundle is never built and every assertion below is vacuous"
        );
        let out_space = space.with_two_s(two_k);
        let h_out = generic_su2_h::<f64>(out_space);
        let ctx = su2_weingarten_operator(&ham, &x);
        let sctx = Su2SectorCtx::new(&ctx, &h_out, two_k);

        let rm = sctx.dressed.rm.clone();
        let rd = sctx.dressed.rd.clone();
        let rdz_before = sctx.dressed.rdz.clone();
        println!(
            "csa bundle: 2k = {two_k}, dim = {dim}, rm = {:?}, rd = {:?}",
            rm.iter().map(Su2RankGrid::total).collect::<Vec<_>>(),
            rd.iter().map(Su2RankGrid::total).collect::<Vec<_>>()
        );
        assert_ne!(
            rm, rd,
            "ANTI-VACUITY: rm and rd must actually DIFFER at 2k > 0 (the dressed fan is up to \
             three-fold), or the three-table requirement and every grid assertion below is vacuous"
        );

        let e = sctx.envs();
        let (u, v) = &sctx.dressed.frames;
        assert_eq!(u.space, out_space, "the masked restamp's stamp");
        assert_eq!(v.space, out_space, "the wire-dressed frame's stamp");
        assert_eq!(u.ranks, rm, "u.ranks must BE rm");
        assert_eq!(v.ranks, rd, "v.ranks must BE rd");

        assert_eq!(e.l.len(), d, "one pure-left entry per site");
        assert_eq!(e.r_kappa.len(), d, "one pure-right entry per site");
        for k in 0..d {
            assert_eq!(e.l[k].k, k, "l[{k}] must be a sketch AT BOND {k}");
            assert_eq!(e.l[k].space, out_space, "l[{k}] stamp");
            assert_eq!(
                e.r_kappa[k].k,
                k + 1,
                "r_kappa[{k}] must be a sketch AT BOND {}",
                k + 1
            );
            assert_eq!(e.r_kappa[k].space, out_space, "r_kappa[{k}] stamp");
            assert_eq!(
                e.l[k].row_ranks(),
                rm[k],
                "l[{k}] rows must be rm[{k}] element for element"
            );
            assert_eq!(
                e.l[k].col_ranks(),
                h_out.row_ranks(u.core(k)),
                "l[{k}] columns must be h_out's row-rank fan-in of u_{k} — the axis that pairs \
                 with the ket's row grid at site {k}, and the axis on which an L summed into an \
                 L-tilde is a release-mode wrong number"
            );
            assert_eq!(
                e.r_kappa[k].col_ranks(),
                rd[k + 1],
                "r_kappa[{k}] columns must be rd[{}] element for element",
                k + 1
            );
            assert_eq!(
                e.r_kappa[k].row_ranks(),
                h_out.col_ranks(v.core(k)),
                "r_kappa[{k}] rows must be h_out's col-rank fan-in of v_{k}"
            );
        }
        assert_eq!(
            e.l[0].row_ranks(),
            rm[0],
            "l[0] is the bond-0 identity seed at rm[0]"
        );
        assert_eq!(
            e.r_kappa[d - 1].col_ranks(),
            rd[d],
            "r_kappa[d-1] is the bond-d identity seed IN OUT_SPACE at rd[d] — a seed taken in x's \
             own space lands on (N, 0, Gamma0), which out_space does not admit at bond d, and \
             zeros the whole right stack silently"
        );

        assert_eq!(e.ws_u.len(), d);
        assert_eq!(e.ws_v.len(), d);
        assert_eq!(e.ws_du.len(), d);
        for k in 0..d {
            assert_eq!(
                e.ws_u[k].i_tab,
                h_out.row_ranges(u.core(k)),
                "ws_u[{k}] must be su2_onesite_workspace(h_out, u_{k}) — the KET's tables"
            );
            assert_eq!(e.ws_u[k].j_tab, h_out.col_ranges(u.core(k)));
            assert_eq!(
                e.ws_v[k].i_tab,
                h_out.row_ranges(v.core(k)),
                "ws_v[{k}] must be su2_onesite_workspace(h_out, v_{k})"
            );
            assert_eq!(e.ws_v[k].j_tab, h_out.col_ranges(v.core(k)));
            let decl = crate::su2::core::Su2Core::<f64>::zeros(out_space, k, &rm[k], &rd[k + 1]);
            assert_eq!(
                e.ws_du[k].i_tab,
                h_out.row_ranges(&decl),
                "ws_du[{k}] must key on the DECLARED component shape rm[{k}] x rd[{}], not on any \
                 xi",
                k + 1
            );
            assert_eq!(e.ws_du[k].j_tab, h_out.col_ranges(&decl));
        }
        let du_last = table_width(&e.ws_u[d - 1].j_tab);
        let dv_first = table_width(&e.ws_v[0].i_tab);
        println!(
            "csa bundle: degenerate tables — |ws_u[d-1].j_tab| = {du_last}, |ws_v[0].i_tab| = \
             {dv_first}"
        );
        assert_eq!(
            du_last, 0,
            "ws_u[d-1] is degenerate BY CONSTRUCTION (its column grid rm[d] is zero-width: \
             (N, 0, Gamma0) is not admissible at bond d in out_space). It must be BUILT — it is, \
             or the indexing above would have panicked — and never asserted non-vacuous"
        );
        assert_eq!(
            dv_first, 0,
            "ws_v[0] is degenerate BY CONSTRUCTION (its row grid rd[0] is zero-width: the fan of \
             (0, 0) is (0, 2k))"
        );

        assert_eq!(
            sctx.dressed.rdz, rdz_before,
            "the A-path bundle must not touch rdz — that is the mask-DOWN Z-side grid, \
             read only by the term-B frame-derivative environments"
        );
        assert_eq!(sctx.dressed.rm, rm, "the bundle must not touch rm");
        assert_eq!(sctx.dressed.rd, rd, "the bundle must not touch rd");
        assert_eq!(sctx.dressed.two_s_flux, two_k);
        assert_eq!(sctx.dressed.out_space, out_space);
    }

    #[test]
    fn csa_env_bundle_is_lazy_while_the_dressed_pair_stays_eager() {
        let (space, ham, x) = live_channel_fixture();
        let two_k = 2usize;
        let out_space = space.with_two_s(two_k);
        let h_out = generic_su2_h::<f64>(out_space);
        let ctx = su2_weingarten_operator(&ham, &x);
        let sctx = Su2SectorCtx::new(&ctx, &h_out, two_k);

        assert_eq!(sctx.dressed.two_s_flux, two_k);
        assert_eq!(sctx.dressed.frames.0.space, out_space);
        assert_eq!(sctx.dressed.frames.1.space, out_space);
        assert!(
            sctx.envs.get().is_none(),
            "Su2SectorCtx::new must NOT build the environment bundle — a context on an empty \
             channel is constructed only to report dim() = 0 and must not pay a two-sweep build"
        );
        let (bl, br) = su2_tangent_bases(&x);
        let basis = tangent_basis_shifted(&bl, &br, two_k);
        assert!(!basis.is_empty(), "ANTI-VACUITY: the channel is live");
        assert!(
            sctx.envs.get().is_none(),
            "reading dim()/basis() must not force the bundle either (dim = {})",
            sctx.dim()
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
        let _ = sctx.a_gram(&basis);
        let _ = sctx.grams(&basis);
        let mut rng = seeded(0xC5_A5);
        let y: Su2TtVector<f64> = randomized_su2_state(out_space, 1, 2, &mut rng);
        let _ = sctx.project(&y);
        assert_eq!(
            std::ptr::from_ref(sctx.envs.get().expect("still built")),
            first,
            "a_gram / grams / project must share the ONE bundle a_apply_fused built"
        );
        println!(
            "csa laziness: 2k = {two_k}, dim = {}, basis = {}",
            sctx.dim(),
            basis.len()
        );
    }

    #[test]
    fn csa_a_zero_width_channel_needs_no_early_return() {
        let (space, ham, x) = zero_width_channel_fixture();
        let d = space.d;
        let two_k = 2usize;
        let (bl, br) = su2_tangent_bases(&x);
        let dim = tangent_dim_grids(&space, &bl.ranks, two_k);
        println!(
            "csa zero-width: ranks = {:?}, 2k = {two_k}, dim = {dim}",
            bl.ranks.iter().map(Su2RankGrid::total).collect::<Vec<_>>()
        );
        assert_eq!(
            dim, 0,
            "fixture premise: rank only at the 2s = 0 bond sectors must make the FEASIBLE 2k = 2 \
             channel zero-width (tri(0, 0, 2) fails), or this arm tests a live channel"
        );
        let out_space = space.with_two_s(two_k);
        assert_eq!(out_space.two_s, two_k);
        let h_out = generic_su2_h::<f64>(out_space);
        let ctx = su2_weingarten_operator(&ham, &x);
        let sctx = Su2SectorCtx::new(&ctx, &h_out, two_k);
        assert_eq!(sctx.dim(), 0);
        assert!(
            tangent_basis_shifted(&bl, &br, two_k).is_empty(),
            "a zero-width channel carries no basis"
        );
        assert!(
            sctx.a_gram(&[]).is_empty() && sctx.a_gram_exact(&[]).is_empty(),
            "both A drivers must answer the empty basis with an empty Gram"
        );
        let (a, b) = sctx.grams(&[]);
        assert!(
            a.is_empty() && b.is_empty(),
            "and so must the paired driver"
        );

        let e = sctx.envs();
        assert_eq!(e.l.len(), d);
        assert_eq!(e.r_kappa.len(), d);
        assert_eq!(e.ws_u.len(), d);
        assert_eq!(e.ws_v.len(), d);
        assert_eq!(e.ws_du.len(), d);

        let mut rng = seeded(0xC5_A3);
        let y: Su2TtVector<f64> = randomized_su2_state(out_space, 1, 2, &mut rng);
        let xi = sctx.project(&y);
        let fused = sctx.a_apply_fused(&xi);
        let exact = sctx.a_apply_exact(&xi);
        println!(
            "csa zero-width: xi coeffs = {}, per-component (rows x cols) = {:?}",
            xi.num_coeffs(),
            (0..d)
                .map(|k| (
                    xi.component(k).row_ranks.total(),
                    xi.component(k).col_ranks.total()
                ))
                .collect::<Vec<_>>()
        );
        assert_eq!(fused.two_s_flux(), two_k);
        assert_eq!(fused.rank(), xi.rank(), "declared row grids");
        assert_eq!(fused.rank_col(), xi.rank_col(), "declared column grids");
        assert_eq!(
            fused.num_coeffs(),
            exact.num_coeffs(),
            "coefficient count on a zero-width channel"
        );
        for k in 0..d {
            let (cf, ce) = (fused.component(k), exact.component(k));
            assert_eq!(cf.space, out_space, "component {k} stamp");
            assert_eq!(cf.row_ranks, ce.row_ranks, "component {k} row grid");
            assert_eq!(
                cf.col_ranks, ce.col_ranks,
                "component {k} col grid — the DECLARED shape, which is what a ported abelian \
                 early return would get wrong"
            );
        }
        for i in 0..fused.num_coeffs() {
            assert_eq!(
                fused.coeff(i),
                0.0,
                "coeff {i}: P^(S') is identically zero on a zero-width channel"
            );
        }
    }

    #[test]
    fn csa_regularized_zero_width_channel_reads_out_empty() {
        use crate::solvers::hess::{CurvatureReg, SectorOpts, SectorReadout};
        use crate::su2::deflate::Su2ColumnDeflation;
        let (space, ham, x) = zero_width_channel_fixture();
        let d = space.d;
        let two_k = 2usize;
        let ctx0 = su2_weingarten_operator(&ham, &x);
        let f = 1.5
            * ctx0
                .bond_gauges()
                .iter()
                .flat_map(|g| g.sigma.iter().flat_map(|(_, v)| v.iter().copied()))
                .fold(f64::INFINITY, f64::min);
        assert!(
            !Su2ColumnDeflation::from_gauges(ctx0.bond_gauges(), f).is_identity(),
            "premise: the regularization cuts a multiplet (else curvature_op() is None)"
        );
        let ctx = su2_weingarten_operator(&ham, &x).with_curvature(CurvatureReg::Hard(f));
        let out_space = space.with_two_s(two_k);
        let h_out = generic_su2_h::<f64>(out_space);
        let sctx = Su2SectorCtx::new(&ctx, &h_out, two_k);
        assert_eq!(sctx.dim(), 0, "fixture premise: the channel is zero-width");
        let opts = SectorOpts {
            readout: SectorReadout::Rpa,
            ..SectorOpts::default()
        };
        let resp = sctx.sector_lowest(2, &opts);
        println!(
            "csa regularized zero-width: dim {}, converged {}, applies {}, omega2 {:?}",
            resp.dim,
            resp.converged,
            resp.applies,
            resp.omega2()
        );
        assert!(resp.converged, "route (a): nothing outstanding");
        assert_eq!(
            (resp.applies, resp.iterations),
            (0, 0),
            "route (a): no work"
        );
        assert!(resp.omega2().expect("Rpa").is_empty());
        let mut rng = seeded(0xC5_A6);
        let y: Su2TtVector<f64> = randomized_su2_state(out_space, 1, 2, &mut rng);
        let xi = sctx.project(&y);
        let bx = sctx.b_apply(&xi);
        assert_eq!(bx.two_s_flux(), two_k);
        assert_eq!(bx.rank(), xi.rank(), "declared row grids");
        assert_eq!(bx.rank_col(), xi.rank_col(), "declared column grids");
        assert_eq!(bx.num_coeffs(), xi.num_coeffs());
        for k in 0..d {
            assert_eq!(
                bx.component(k).col_ranks,
                xi.component(k).col_ranks,
                "component {k}"
            );
        }
        for i in 0..bx.num_coeffs() {
            assert_eq!(
                bx.coeff(i),
                0.0,
                "coeff {i}: B on a zero-width channel is zero"
            );
        }
    }
}

#[cfg(test)]
mod t8s_su2_readout_tests {
    use super::Su2SectorResponse;
    use crate::scalar::C64;
    use crate::solvers::hess::{
        MetricEvent, RpaMetric, RpaMode, RpaModeKind, RpaReadout, RpaReport, SectorSpectrum,
    };
    use crate::su2::qn::Su2Space;

    struct Report {
        n_neg_basis: usize,
        general_path: bool,
        complex: Vec<(f64, f64)>,
    }

    fn signed(omega2: Vec<f64>, signature: Vec<i8>, report: Option<Report>) -> Su2SectorResponse {
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
        let ev = |accepted: bool| MetricEvent {
            basis_len: 3,
            gn_before: 1.0,
            gn_after: 0.9,
            wnorm2: if accepted { -0.2 } else { -4e-7 },
            accepted,
        };
        let (events, n_negative_basis, general_path) = match &report {
            None => (Vec::new(), 0, false),
            Some(r) if r.n_neg_basis == 0 => (vec![ev(false)], 0, r.general_path),
            Some(r) => (
                (0..r.n_neg_basis).map(|_| ev(true)).collect(),
                r.n_neg_basis,
                r.general_path,
            ),
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
        Su2SectorResponse {
            spectrum: SectorSpectrum::Rpa(RpaReadout {
                passed_over: Vec::new(),
                modes,
                converged: true,
                iterations: 4,
                applies: 9,
                report: RpaReport::Signed {
                    metric: RpaMetric::Ii,
                    events,
                    n_negative_basis,
                    general_path,
                    diag: None,
                },
            }),
            dim: 4,
            two_s_out: 2,
            space: Su2Space::new(4, 4, 2),
            converged: true,
            iterations: 4,
            applies: 9,
            sigma_min: 0.5,
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
    fn t8s_su2_omega_refuses_a_flagged_solve() {
        let flag_only = signed(
            vec![2.0, 12.0],
            vec![1, 1],
            Some(report(0, false, Vec::new())),
        );
        assert!(flag_only.metric_indefinite());
        assert_eq!(flag_only.omega2(), Some(vec![2.0, 12.0]));
        assert!(
            flag_only.omega().is_none(),
            "a flagged solve's converged does not license its omega (Q5)"
        );
        let held = signed(
            vec![2.0, 12.0],
            vec![1, -1],
            Some(report(1, false, Vec::new())),
        );
        assert!(held.omega().is_none(), "flagged + s = -1");
        let sig_only = signed(vec![2.0, 12.0], vec![1, -1], None);
        assert!(!sig_only.metric_indefinite());
        assert!(
            sig_only.omega().is_none(),
            "the s = -1 clause must hold on its own (a report-blind rule would publish |a| here)"
        );
        let clean = signed(vec![2.0, 12.0], vec![1, 1], None);
        let w = clean
            .omega()
            .expect("unflagged, all positive, all s = +1, no witness");
        assert!(
            (w[0] - 2.0f64.sqrt()).abs() <= 1e-15 && (w[1] - 12.0f64.sqrt()).abs() <= 1e-15,
            "{w:?}"
        );
    }

    #[test]
    fn t8s_su2_index_lower_bound_counts_the_inertia() {
        let five = signed(
            vec![-1.0, 2.0, 3.0],
            vec![-1, -1, 1],
            Some(report(2, true, vec![pair(1.5, 0.3)])),
        );
        assert_eq!(five.index_lower_bound(), Some(5));
        let eight = signed(
            vec![-2.0, -1.0, 0.5, 3.0],
            vec![-1, 1, -1, 1],
            Some(report(2, true, vec![pair(1.5, 0.3), pair(2.5, 0.1)])),
        );
        assert_eq!(eight.index_lower_bound(), Some(8));
        let strict = signed(
            vec![0.0, 1.0],
            vec![-1, 1],
            Some(report(1, false, Vec::new())),
        );
        assert_eq!(
            strict.index_lower_bound(),
            Some(0),
            "omega^2 > 0 is strict: a zero entry carries no inertia"
        );
        assert_eq!(
            signed(Vec::new(), Vec::new(), None).index_lower_bound(),
            Some(0)
        );
        let tda = Su2SectorResponse {
            spectrum: SectorSpectrum::Tda {
                theta: vec![-0.5, 0.2],
                negative: Vec::new(),
            },
            ..signed(Vec::new(), Vec::new(), None)
        };
        assert_eq!(
            tda.index_lower_bound(),
            None,
            "eig(A^S') has no pencil inertia"
        );
    }
}

#[cfg(test)]
mod d0_tests;
