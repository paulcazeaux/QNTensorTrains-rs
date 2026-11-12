// Copyright (C) 2023-2026 Paul Cazeaux
//
// This file is part of qntt. qntt is free software: you can
// redistribute it and/or modify it under the terms of the GNU
// General Public License as published by the Free Software
// Foundation, either version 3 of the License, or (at your
// option) any later version. See LICENSE for details.

//! The `Layer` abstraction: everything the C API needs from a symmetry layer, implemented once for
//! the abelian `(Nup, Ndn)` stack and once for the SU(2) stack.

use rand_xoshiro::Xoshiro256PlusPlus;

use qntt::hamiltonian::Construction;
use qntt::solvers::als::AlsOptions;
use qntt::solvers::excited::ExcitedInfo;
use qntt::solvers::mals::{MalsHistory, MalsOptions};
use qntt::solvers::multistate::{SaHistory, StateAverageOptions};
use qntt::solvers::ssmals::{SsHistory, StateSpecificOptions};
use qntt::solvers::tdvp::TdvpOptions;
use qntt::solvers::{TtHamiltonian, TtState};
use qntt::{CoeffTensor, Scalar, ThreeBody};

use crate::err::Error;
use crate::handle::{HandleType, ScalarTag};
use crate::opts::{
    ExcitedSchedules, QnttExcitedOptions, QnttRiemannianOpts, excited_opts_abelian,
    excited_opts_su2,
};

fn pooled_descending(vals: impl Iterator<Item = (f64, f64)>) -> Vec<(f64, f64)> {
    let mut v: Vec<(f64, f64)> = vals.collect();
    v.sort_by(|a, b| b.0.total_cmp(&a.0));
    v
}

fn rayleigh_of<L: Layer>(h: &L::Ham, x: &L::Vector) -> Result<f64, Error> {
    let n = x.norm();
    if n == 0.0 {
        return Err(Error::invalid(
            "the Riemannian gradient is not defined at the zero train",
        ));
    }
    Ok(h.rayleigh_quotient(x).re_s() / (n * n))
}

/// One symmetry layer at one scalar type.
pub trait Layer: Sized + 'static {
    /// `f64` or `C64`.
    type T: ScalarTag;
    /// `QnSpace` / `Su2Space` — small, `Copy`, carried by value everywhere.
    type Space: Copy + Send + Sync + 'static;
    /// `Sector` / `Su2Sector` — a pair of non-negative integers.
    type Sector: Copy;
    /// `RankGrid` / `Su2RankGrid` — a dense grid over the sector rectangle.
    type Grid: Clone;
    /// The layer's block-family enum. The two layers have DIFFERENT ones: abelian `Up`/`Dn` are
    /// electron spins, SU(2) `Up`/`Dn` are the `S ± ½` genealogical coupling branches (see
    /// `src/su2/mod.rs`). The wire codes `0..=3` are the same; the meaning is not.
    type Fam: Copy;
    /// `TtVector<T>` / `Su2TtVector<T>`.
    type Vector: TtState<Scalar = Self::T> + HandleType + Send + Sync + 'static;
    /// `SparseHamiltonian<T>` / `Su2SparseHamiltonian<T>`.
    type Ham: TtHamiltonian<Self::Vector> + HandleType + Send + Sync + 'static;

    /// Build the space descriptor. `a`/`b` are `(Nup, Ndn)` for the abelian layer and `(N, 2S)`
    /// for SU(2). Errors (rather than panicking) on out-of-range arguments so a driver typo is a
    /// clean exception.
    fn space_new(d: usize, a: usize, b: usize) -> Result<Self::Space, Error>;
    /// `(d, a, b)` — the inverse of [`Layer::space_new`].
    fn space_parts(space: &Self::Space) -> (usize, usize, usize);
    /// Extent of the rank/sector grid, second axis fastest in the flat layout.
    fn grid_extent(space: &Self::Space) -> (usize, usize);
    /// Extent of the sector grid's irrep axis: `|G|` on either layer (`space.pg.order()`), and `1`
    /// exactly when the space carries no point group. The legacy two-axis marshalling in `api.rs`
    /// is only correct at `1`, so this is what gates it (Rule L).
    fn space_nirrep(space: &Self::Space) -> usize;
    /// Admissible sectors at bond `b`, in the layer's canonical order (the order every block
    /// iteration in the core crate uses).
    fn bond_sectors(space: &Self::Space, b: usize) -> Vec<Self::Sector>;
    /// Per-bond grids of the maximal admissible ranks (the full-rank manifold).
    fn max_ranks(space: &Self::Space) -> Vec<Self::Grid>;

    /// Block family from its wire code `0..=3`.
    fn family(v: i32) -> Result<Self::Fam, Error>;
    fn sector_new(a: usize, b: usize) -> Self::Sector;
    fn sector_parts(s: Self::Sector) -> (usize, usize);
    fn grid_zeros(space: &Self::Space) -> Self::Grid;
    fn grid_get(g: &Self::Grid, s: Self::Sector) -> usize;
    fn grid_set(g: &mut Self::Grid, s: Self::Sector, r: usize);

    fn vec_zeros(space: Self::Space) -> Self::Vector;
    /// All-zero blocks at prescribed per-bond ranks (`ranks.len() == d + 1`).
    fn vec_from_ranks(space: Self::Space, ranks: &[Self::Grid]) -> Self::Vector;
    fn vec_randn(
        space: Self::Space,
        ranks: &[Self::Grid],
        rng: &mut Xoshiro256PlusPlus,
    ) -> Self::Vector;
    fn vec_space(v: &Self::Vector) -> Self::Space;
    fn vec_ranks(v: &Self::Vector) -> &[Self::Grid];
    /// `Some(k)` when the train is orthogonal with its center at core `k`.
    fn vec_ortho(v: &Self::Vector) -> Option<usize>;
    fn vec_check(v: &Self::Vector) -> bool;
    /// Dimensions of one core block, or `None` when the block is absent.
    fn vec_block_dims(
        v: &Self::Vector,
        k: usize,
        f: Self::Fam,
        s: Self::Sector,
    ) -> Option<(usize, usize)>;
    /// Copy one core block out, column-major.
    fn vec_block_copy_out(
        v: &Self::Vector,
        k: usize,
        f: Self::Fam,
        s: Self::Sector,
        dst: &mut [Self::T],
    ) -> Option<(usize, usize)>;
    /// Overwrite one core block from a column-major buffer.
    fn vec_block_copy_in(
        v: &mut Self::Vector,
        k: usize,
        f: Self::Fam,
        s: Self::Sector,
        src: &[Self::T],
    ) -> Option<(usize, usize)>;

    /// In-place complex conjugation; the identity for real `T`.
    fn vec_conj_mut(x: &mut Self::Vector);

    /// Per-interior-bond Schmidt data (`d - 1` entries, bonds `1..d`): `(σ, weight)` pairs pooled
    /// over the bond's sectors and sorted descending in `σ`.
    fn bond_schmidt(x: &Self::Vector) -> Vec<Vec<(f64, f64)>>;

    /// Documentation anchor for the one sum rule both layers obey. At every interior bond of a
    /// unit-norm state,
    const SCHMIDT_SUM_RULE: () = ();

    /// `‖P_T(H x − λ x)‖` with `λ` the Rayleigh quotient at `x` — the Riemannian gradient norm,
    /// i.e. the criticality gate (the RPA reference criterion is `< 1e-9`).
    fn grad_norm(h: &Self::Ham, x: &Self::Vector) -> Result<f64, Error>;

    /// Write the train to `path` in the `QNTT-TT` v1 format (`src/tt_io.rs`), truncating.
    fn vec_save(x: &Self::Vector, path: &std::path::Path) -> Result<(), qntt::tt_io::TtIoError>;
    /// Read a train written by [`Layer::vec_save`]. The file's own layer and scalar tags are
    /// checked against `Self`, so loading an SU(2) file through an abelian symbol (or an `f64`
    /// file through a `C64` one) is a clean error, never a reinterpretation.
    fn vec_load(path: &std::path::Path) -> Result<Self::Vector, qntt::tt_io::TtIoError>;

    /// `construction` is honoured by the abelian layer; the SU(2) assembly has a single
    /// construction and rejects a non-default request.
    fn ham_new(
        space: Self::Space,
        t: &CoeffTensor<Self::T>,
        v: &CoeffTensor<Self::T>,
        w: Option<&dyn ThreeBody<Self::T>>,
        eps: f64,
        construction: Option<Construction>,
    ) -> Result<Self::Ham, Error>;
    fn ham_space(h: &Self::Ham) -> Self::Space;
    /// `⟨x, H·y⟩` (no normalization).
    fn xt_h_y(x: &Self::Vector, h: &Self::Ham, y: &Self::Vector) -> Self::T;
    /// Summed-over-sector automaton bond dimensions, one per bond, when the layer reports them.
    fn ham_bond_dims(h: &Self::Ham) -> Option<Vec<usize>>;

    fn als(h: &Self::Ham, x0: &Self::Vector, opts: &AlsOptions) -> (Self::T, Self::Vector);
    /// Returns the per-iteration history alongside the state; the final energy is `history.lambda`.
    fn mals(h: &Self::Ham, x0: &Self::Vector, opts: &MalsOptions) -> (MalsHistory, Self::Vector);
    fn tdvp(h: &Self::Ham, x0: &Self::Vector, opts: &TdvpOptions) -> (f64, Self::Vector);
    /// Riemannian optimization (`cg = false` ⇒ steepest descent). The two layers have separate
    /// (field-identical) option structs, so the C mirror is converted inside each impl.
    fn riemannian(
        h: &Self::Ham,
        x0: &Self::Vector,
        opts: &QnttRiemannianOpts,
        cg: bool,
    ) -> Result<(f64, Self::Vector), Error>;

    /// The `nroots` lowest states sharing one train. Returns the per-iteration history alongside
    /// the states; the converged energies are `history.lambdas`, in label order.
    fn state_average_mals(
        h: &Self::Ham,
        x0: &Self::Vector,
        nroots: usize,
        opts: &StateAverageOptions,
    ) -> (SaHistory, Vec<Self::Vector>);

    /// One state against fixed `lower` states, by penalty or projection. `lower` is never
    /// mutated and need not be normalized; `&[]` with `opts.roots == 1` is the delegate path,
    /// bitwise `mals` (at `roots > 1` the block sweep runs even with no lower states).
    fn state_specific_mals(
        h: &Self::Ham,
        x0: &Self::Vector,
        lower: &[Self::Vector],
        opts: &StateSpecificOptions,
    ) -> (SsHistory, Self::Vector);

    /// The excited-state ladder. Builds this layer's `ExcitedOpts::new(&vec_space(x0))`, overwrites
    /// it from the C mirror and the schedules (so the space-derived CG block stays as `new` set
    /// it), and runs the four stages.
    #[allow(clippy::type_complexity)]
    fn excited_states(
        h: &Self::Ham,
        x0: &Self::Vector,
        lower: &[Self::Vector],
        opts: &QnttExcitedOptions,
        sched: &ExcitedSchedules<'_>,
    ) -> Result<(Vec<f64>, Vec<Self::Vector>, ExcitedInfo<Self::Vector>), Error>;
}

pub struct Abelian<T: ScalarTag>(std::marker::PhantomData<T>);

impl<T: ScalarTag> Layer for Abelian<T> {
    type T = T;
    type Space = qntt::QnSpace;
    type Sector = qntt::Sector;
    type Grid = qntt::RankGrid;
    type Fam = qntt::core::Family;
    type Vector = qntt::TtVector<T>;
    type Ham = qntt::SparseHamiltonian<T>;

    fn space_new(d: usize, a: usize, b: usize) -> Result<Self::Space, Error> {
        if a > d || b > d {
            return Err(Error::invalid(format!(
                "abelian space: need Nup ({a}) <= d ({d}) and Ndn ({b}) <= d"
            )));
        }
        Ok(qntt::QnSpace::new(d, a, b))
    }

    fn space_parts(space: &Self::Space) -> (usize, usize, usize) {
        (space.d, space.nup, space.ndn)
    }

    fn grid_extent(space: &Self::Space) -> (usize, usize) {
        (space.nup + 1, space.ndn + 1)
    }

    fn space_nirrep(space: &Self::Space) -> usize {
        space.pg.order() as usize
    }

    fn bond_sectors(space: &Self::Space, b: usize) -> Vec<Self::Sector> {
        space.bond_sectors(b)
    }

    fn max_ranks(space: &Self::Space) -> Vec<Self::Grid> {
        qntt::max_ranks(space)
    }

    fn family(v: i32) -> Result<Self::Fam, Error> {
        crate::enums::family(v)
    }

    fn sector_new(a: usize, b: usize) -> Self::Sector {
        qntt::Sector::new(a, b)
    }

    fn sector_parts(s: Self::Sector) -> (usize, usize) {
        (s.nup, s.ndn)
    }

    fn grid_zeros(space: &Self::Space) -> Self::Grid {
        qntt::RankGrid::zeros(space)
    }

    fn grid_get(g: &Self::Grid, s: Self::Sector) -> usize {
        g[s]
    }

    fn grid_set(g: &mut Self::Grid, s: Self::Sector, r: usize) {
        g[s] = r;
    }

    fn vec_zeros(space: Self::Space) -> Self::Vector {
        qntt::tt_zeros(space)
    }

    fn vec_from_ranks(space: Self::Space, ranks: &[Self::Grid]) -> Self::Vector {
        let cores = (0..space.d)
            .map(|k| qntt::core::SparseCore::zeros(space, k, &ranks[k], &ranks[k + 1]))
            .collect();
        qntt::TtVector::new(space, cores)
    }

    fn vec_randn(
        space: Self::Space,
        ranks: &[Self::Grid],
        rng: &mut Xoshiro256PlusPlus,
    ) -> Self::Vector {
        qntt::tt_randn(space, ranks, rng)
    }

    fn vec_space(v: &Self::Vector) -> Self::Space {
        v.space
    }

    fn vec_ranks(v: &Self::Vector) -> &[Self::Grid] {
        &v.ranks
    }

    fn vec_ortho(v: &Self::Vector) -> Option<usize> {
        v.ortho
    }

    fn vec_check(v: &Self::Vector) -> bool {
        v.check()
    }

    fn vec_block_dims(
        v: &Self::Vector,
        k: usize,
        f: Self::Fam,
        s: Self::Sector,
    ) -> Option<(usize, usize)> {
        v.cores[k].block_dims(f, s)
    }

    fn vec_block_copy_out(
        v: &Self::Vector,
        k: usize,
        f: Self::Fam,
        s: Self::Sector,
        dst: &mut [Self::T],
    ) -> Option<(usize, usize)> {
        v.cores[k].block_copy_out(f, s, dst)
    }

    fn vec_block_copy_in(
        v: &mut Self::Vector,
        k: usize,
        f: Self::Fam,
        s: Self::Sector,
        src: &[Self::T],
    ) -> Option<(usize, usize)> {
        v.cores[k].block_copy_in(f, s, src)
    }

    fn vec_conj_mut(x: &mut Self::Vector) {
        x.conj_mut();
    }

    fn bond_schmidt(x: &Self::Vector) -> Vec<Vec<(f64, f64)>> {
        let (bl, br) = qntt::solvers::riemannian::tangent_bases(x);
        let floor = qntt::core::weingarten::FloorPolicy::default_hard();
        qntt::core::weingarten::bond_gauges(&bl, &br, floor)
            .iter()
            .map(|g| {
                pooled_descending(
                    g.sigma
                        .iter()
                        .flat_map(|(_, sv)| sv.iter().map(|&v| (v, 1.0))),
                )
            })
            .collect()
    }

    fn grad_norm(h: &Self::Ham, x: &Self::Vector) -> Result<f64, Error> {
        let lambda = rayleigh_of::<Self>(h, x)?;
        let (bl, br) = qntt::solvers::riemannian::tangent_bases(x);
        Ok(qntt::solvers::hess::riemannian_gradient_env(h, &bl, &br, lambda).norm())
    }

    fn vec_save(x: &Self::Vector, path: &std::path::Path) -> Result<(), qntt::tt_io::TtIoError> {
        qntt::tt_io::save_tt(path, x)
    }

    fn vec_load(path: &std::path::Path) -> Result<Self::Vector, qntt::tt_io::TtIoError> {
        qntt::tt_io::load_tt(path)
    }

    fn ham_new(
        space: Self::Space,
        t: &CoeffTensor<T>,
        v: &CoeffTensor<T>,
        w: Option<&dyn ThreeBody<T>>,
        eps: f64,
        construction: Option<Construction>,
    ) -> Result<Self::Ham, Error> {
        Ok(match construction {
            None => qntt::SparseHamiltonian::new(space, t, v, w, eps),
            Some(c) => qntt::SparseHamiltonian::new_with(space, t, v, w, eps, c),
        })
    }

    fn ham_space(h: &Self::Ham) -> Self::Space {
        h.space
    }

    fn xt_h_y(x: &Self::Vector, h: &Self::Ham, y: &Self::Vector) -> T {
        qntt::xt_h_y(x, h, y)
    }

    fn ham_bond_dims(h: &Self::Ham) -> Option<Vec<usize>> {
        Some(h.automaton_bond_dims())
    }

    fn als(h: &Self::Ham, x0: &Self::Vector, opts: &AlsOptions) -> (T, Self::Vector) {
        qntt::solvers::als::als(h, x0, opts)
    }

    fn mals(h: &Self::Ham, x0: &Self::Vector, opts: &MalsOptions) -> (MalsHistory, Self::Vector) {
        qntt::solvers::mals::mals_with_history(h, x0, opts)
    }

    fn tdvp(h: &Self::Ham, x0: &Self::Vector, opts: &TdvpOptions) -> (f64, Self::Vector) {
        qntt::solvers::tdvp::tdvp(h, x0, opts)
    }

    fn riemannian(
        h: &Self::Ham,
        x0: &Self::Vector,
        opts: &QnttRiemannianOpts,
        cg: bool,
    ) -> Result<(f64, Self::Vector), Error> {
        let o = qntt::solvers::riemannian::RiemannianOpts {
            eps: opts.eps,
            max_iter: opts.max_iter,
            gradtol: opts.gradtol,
            seed: opts.seed,
            variant: crate::enums::cg_variant(opts.variant)?,
            preserve_grid: false,
            retraction: qntt::solvers::riemannian::Retraction::Hosvd,
        };
        Ok(if cg {
            qntt::solvers::riemannian::riemannian_cg(h, x0, &o)
        } else {
            qntt::solvers::riemannian::riemannian_gd(h, x0, &o)
        })
    }

    fn state_average_mals(
        h: &Self::Ham,
        x0: &Self::Vector,
        nroots: usize,
        opts: &StateAverageOptions,
    ) -> (SaHistory, Vec<Self::Vector>) {
        qntt::solvers::multistate::state_average_mals_with_history(h, x0, nroots, opts)
    }

    fn state_specific_mals(
        h: &Self::Ham,
        x0: &Self::Vector,
        lower: &[Self::Vector],
        opts: &StateSpecificOptions,
    ) -> (SsHistory, Self::Vector) {
        qntt::solvers::ssmals::state_specific_mals_with_history(h, x0, lower, opts)
    }

    fn excited_states(
        h: &Self::Ham,
        x0: &Self::Vector,
        lower: &[Self::Vector],
        opts: &QnttExcitedOptions,
        sched: &ExcitedSchedules<'_>,
    ) -> Result<(Vec<f64>, Vec<Self::Vector>, ExcitedInfo<Self::Vector>), Error> {
        let o = excited_opts_abelian(
            opts,
            sched,
            qntt::solvers::excited::ExcitedOpts::new(&Self::vec_space(x0)),
        )?;
        Ok(qntt::solvers::excited::excited_states(h, x0, lower, &o))
    }
}

pub struct Su2<T: ScalarTag>(std::marker::PhantomData<T>);

impl<T: ScalarTag> Layer for Su2<T> {
    type T = T;
    type Space = qntt::su2::qn::Su2Space;
    type Sector = qntt::su2::qn::Su2Sector;
    type Grid = qntt::su2::qn::Su2RankGrid;
    type Fam = qntt::su2::core::Family;
    type Vector = qntt::su2::vector::Su2TtVector<T>;
    type Ham = qntt::su2::hamiltonian::Su2SparseHamiltonian<T>;

    fn space_new(d: usize, a: usize, b: usize) -> Result<Self::Space, Error> {
        if a > 2 * d {
            return Err(Error::invalid(format!(
                "SU(2) space: need N ({a}) <= 2d ({})",
                2 * d
            )));
        }
        if b > a || !(a - b).is_multiple_of(2) {
            return Err(Error::invalid(format!(
                "SU(2) space: need 2S ({b}) <= N ({a}) with N - 2S even"
            )));
        }
        Ok(qntt::su2::qn::Su2Space::new(d, a, b))
    }

    fn space_parts(space: &Self::Space) -> (usize, usize, usize) {
        (space.d, space.n, space.two_s)
    }

    fn grid_extent(space: &Self::Space) -> (usize, usize) {
        (space.n + 1, space.n + 1)
    }

    fn space_nirrep(space: &Self::Space) -> usize {
        space.pg.order() as usize
    }

    fn bond_sectors(space: &Self::Space, b: usize) -> Vec<Self::Sector> {
        space.bond_sectors(b)
    }

    fn max_ranks(space: &Self::Space) -> Vec<Self::Grid> {
        qntt::su2::qn::max_ranks(space)
    }

    fn family(v: i32) -> Result<Self::Fam, Error> {
        crate::su2_pointgroup::su2_family(v)
    }

    fn sector_new(a: usize, b: usize) -> Self::Sector {
        qntt::su2::qn::Su2Sector::new(a, b)
    }

    fn sector_parts(s: Self::Sector) -> (usize, usize) {
        (s.n, s.two_s)
    }

    fn grid_zeros(space: &Self::Space) -> Self::Grid {
        qntt::su2::qn::Su2RankGrid::zeros(space)
    }

    fn grid_get(g: &Self::Grid, s: Self::Sector) -> usize {
        g[s]
    }

    fn grid_set(g: &mut Self::Grid, s: Self::Sector, r: usize) {
        g[s] = r;
    }

    fn vec_zeros(space: Self::Space) -> Self::Vector {
        qntt::su2::vector::tt_zeros(space)
    }

    fn vec_from_ranks(space: Self::Space, ranks: &[Self::Grid]) -> Self::Vector {
        let cores = (0..space.d)
            .map(|k| qntt::su2::core::Su2Core::zeros(space, k, &ranks[k], &ranks[k + 1]))
            .collect();
        qntt::su2::vector::Su2TtVector::new(space, cores)
    }

    fn vec_randn(
        space: Self::Space,
        ranks: &[Self::Grid],
        rng: &mut Xoshiro256PlusPlus,
    ) -> Self::Vector {
        qntt::su2::vector::tt_randn(space, ranks, rng)
    }

    fn vec_space(v: &Self::Vector) -> Self::Space {
        v.space
    }

    fn vec_ranks(v: &Self::Vector) -> &[Self::Grid] {
        &v.ranks
    }

    fn vec_ortho(v: &Self::Vector) -> Option<usize> {
        v.ortho
    }

    fn vec_check(v: &Self::Vector) -> bool {
        v.check()
    }

    fn vec_block_dims(
        v: &Self::Vector,
        k: usize,
        f: Self::Fam,
        s: Self::Sector,
    ) -> Option<(usize, usize)> {
        v.cores[k].block_dims(f, s)
    }

    fn vec_block_copy_out(
        v: &Self::Vector,
        k: usize,
        f: Self::Fam,
        s: Self::Sector,
        dst: &mut [Self::T],
    ) -> Option<(usize, usize)> {
        v.cores[k].block_copy_out(f, s, dst)
    }

    fn vec_block_copy_in(
        v: &mut Self::Vector,
        k: usize,
        f: Self::Fam,
        s: Self::Sector,
        src: &[Self::T],
    ) -> Option<(usize, usize)> {
        v.cores[k].block_copy_in(f, s, src)
    }

    fn vec_conj_mut(x: &mut Self::Vector) {
        x.conj_mut();
    }

    fn bond_schmidt(x: &Self::Vector) -> Vec<Vec<(f64, f64)>> {
        let (bl, br) = qntt::su2::riemannian::tangent_bases(x);
        let floor = qntt::core::weingarten::FloorPolicy::default_hard();
        let two_s_total = x.space.two_s;
        qntt::su2::weingarten::bond_gauges(&bl, &br, floor)
            .iter()
            .map(|g| {
                pooled_descending(g.sigma.iter().flat_map(|(s, sv)| {
                    let w = qntt::su2::centerweight(s.two_s, two_s_total);
                    sv.iter().map(move |&v| (v, w))
                }))
            })
            .collect()
    }

    fn grad_norm(h: &Self::Ham, x: &Self::Vector) -> Result<f64, Error> {
        let lambda = rayleigh_of::<Self>(h, x)?;
        let (bl, br) = qntt::su2::riemannian::tangent_bases(x);
        Ok(qntt::su2::hess::riemannian_gradient_env(h, &bl, &br, lambda).norm())
    }

    fn vec_save(x: &Self::Vector, path: &std::path::Path) -> Result<(), qntt::tt_io::TtIoError> {
        qntt::tt_io::save_su2_tt(path, x)
    }

    fn vec_load(path: &std::path::Path) -> Result<Self::Vector, qntt::tt_io::TtIoError> {
        qntt::tt_io::load_su2_tt(path)
    }

    fn ham_new(
        space: Self::Space,
        t: &CoeffTensor<T>,
        v: &CoeffTensor<T>,
        w: Option<&dyn ThreeBody<T>>,
        eps: f64,
        construction: Option<Construction>,
    ) -> Result<Self::Ham, Error> {
        if construction.is_some() {
            return Err(Error::invalid(
                "the SU(2) layer has a single operator construction; pass the default",
            ));
        }
        Ok(qntt::su2::hamiltonian::Su2SparseHamiltonian::new(
            space, t, v, w, eps,
        ))
    }

    fn ham_space(h: &Self::Ham) -> Self::Space {
        h.space
    }

    fn xt_h_y(x: &Self::Vector, h: &Self::Ham, y: &Self::Vector) -> T {
        qntt::su2::hamiltonian::xt_h_y(x, h, y)
    }

    fn ham_bond_dims(_h: &Self::Ham) -> Option<Vec<usize>> {
        None
    }

    fn als(h: &Self::Ham, x0: &Self::Vector, opts: &AlsOptions) -> (T, Self::Vector) {
        qntt::su2::solvers::als::als(h, x0, opts)
    }

    fn mals(h: &Self::Ham, x0: &Self::Vector, opts: &MalsOptions) -> (MalsHistory, Self::Vector) {
        qntt::su2::solvers::mals::mals_with_history(h, x0, opts)
    }

    fn tdvp(h: &Self::Ham, x0: &Self::Vector, opts: &TdvpOptions) -> (f64, Self::Vector) {
        qntt::su2::solvers::tdvp::tdvp(h, x0, opts)
    }

    fn riemannian(
        h: &Self::Ham,
        x0: &Self::Vector,
        opts: &QnttRiemannianOpts,
        cg: bool,
    ) -> Result<(f64, Self::Vector), Error> {
        let variant = crate::enums::su2_cg_variant(opts.variant)?;
        let o = qntt::su2::riemannian::RiemannianOpts {
            eps: opts.eps,
            max_iter: opts.max_iter,
            gradtol: opts.gradtol,
            seed: opts.seed,
            variant,
            preserve_grid: false,
        };
        Ok(if cg {
            qntt::su2::riemannian::riemannian_cg(h, x0, &o)
        } else {
            qntt::su2::riemannian::riemannian_gd(h, x0, &o)
        })
    }

    fn state_average_mals(
        h: &Self::Ham,
        x0: &Self::Vector,
        nroots: usize,
        opts: &StateAverageOptions,
    ) -> (SaHistory, Vec<Self::Vector>) {
        qntt::su2::solvers::multistate::state_average_mals_with_history(h, x0, nroots, opts)
    }

    fn state_specific_mals(
        h: &Self::Ham,
        x0: &Self::Vector,
        lower: &[Self::Vector],
        opts: &StateSpecificOptions,
    ) -> (SsHistory, Self::Vector) {
        qntt::su2::solvers::ssmals::state_specific_mals_with_history(h, x0, lower, opts)
    }

    fn excited_states(
        h: &Self::Ham,
        x0: &Self::Vector,
        lower: &[Self::Vector],
        opts: &QnttExcitedOptions,
        sched: &ExcitedSchedules<'_>,
    ) -> Result<(Vec<f64>, Vec<Self::Vector>, ExcitedInfo<Self::Vector>), Error> {
        let o = excited_opts_su2(
            opts,
            sched,
            qntt::su2::solvers::excited::ExcitedOpts::new(&Self::vec_space(x0)),
        )?;
        Ok(qntt::su2::solvers::excited::excited_states(
            h, x0, lower, &o,
        ))
    }
}
