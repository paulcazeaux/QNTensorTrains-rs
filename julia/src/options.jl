# Copyright (C) 2023-2026 Paul Cazeaux
#
# This file is part of qntt. qntt is free software: you can
# redistribute it and/or modify it under the terms of the GNU
# General Public License as published by the Free Software
# Foundation, either version 3 of the License, or (at your
# option) any later version. See LICENSE for details.

# `isbits` mirrors of the C option structs (`capi/src/opts.rs`). Field order and types are ABI —
# they must match the Rust `#[repr(C)]` layouts exactly. Each has a `defaults()` that asks the
# library rather than hard-coding, so the Julia keyword defaults can never drift from the core
# crate's.

"""ALS (one-site DMRG) options; see [`als`](@ref)."""
struct AlsOptions
    eps::Float64
    max_iter::Csize_t
    verbose::Cint
end

"""MALS (two-site DMRG) options; the rank and noise schedules are passed separately. See
[`mals`](@ref)."""
struct MalsOptions
    eps::Float64
    max_iter::Csize_t
    verbose::Cint
    seed::UInt64
    noise_type::Cint     # 0 = random, 1 = perturbative (White density-matrix noise, the default)
    conv_tol::Float64    # relative energy-improvement stop; 0 = use eps (the historical behaviour)
end

"""One outer iteration's diagnostics, as returned by [`mals`](@ref) in its `history`."""
struct MalsSweep
    energy::Float64
    discarded_weight::Float64
    max_bond_dim::Csize_t    # largest pooled total bond dimension reached in this sweep
    bond_dim::Csize_t        # the pooled cap in force this iteration (0 = uncapped)
    noise::Float64
end

"""Imaginary-time TDVP options; see [`tdvp`](@ref).

`chi` is a pooled total bond-dimension cap (SU(2): reduced multiplets); `0` = none."""
struct TdvpOptions
    dtau::Float64
    eps::Float64
    trunc::Float64
    max_sweeps::Csize_t
    chi::Csize_t
    exptol::Float64
    verbose::Cint
end

"""Rounded-Lanczos options; see [`lanczos`](@ref)."""
struct LanczosOptions
    tol::Float64
    max_iter::Csize_t
end

"""Riemannian optimization options; see [`riemannian`](@ref)."""
struct RiemannianOptions
    eps::Float64
    max_iter::Csize_t
    gradtol::Float64
    seed::UInt64
    variant::Cint
end

"""Matrix-free RPA (symplectic) options; see [`rpa_signed`](@ref) and [`rpa`](@ref). `max_iter`
is the budget in PAIR-EQUIVALENTS (the paired Davidson builds at most `max_iter` `(A, B)` apply
pairs, the signed W-metric kernel `⌈max_iter/2⌉` basis vectors of two applies each; default 800);
`tol` is relative in `ω` units (pencil residual `‖Hz − ωΣz‖/‖z‖ ≤ tol·ω_max`, or `tol·|ω|` for an
imaginary or complex mode; default 1e-8)."""
struct SymplecticOptions
    max_iter::Csize_t
    tol::Float64
end

"""The exposed knobs of a cross-sector (shifted-channel) readout; see [`sector_rpa`](@ref).
`readout`, `metric` and `kernel` are wire codes — pass them as the Symbols
[`QNTT.SECTOR_READOUTS`](@ref), [`QNTT.RPA_METRIC_CODES`](@ref) and [`QNTT.RPA_KERNEL_CODES`](@ref)
name and `QNTT.setfields` converts them. `seed` is a call parameter of [`sector_rpa`](@ref), not a
field. `max_iter` counts outer iterations on `:tda` and pair-equivalents on `:rpa` (`(A, B)` apply
pairs: `max_iter` basis vectors on `:paired`, `⌈max_iter/2⌉` of two applies each on `:signed`).
`max_subspace` bounds the `:rpa` readout's live Davidson basis (`0` = the library's auto bound
`max(8(k+b), 32)`, clamped to the channel dimension; too tight a bound limit-cycles on hard
spectra); `:tda` ignores it."""
struct SectorOptions
    max_iter::Csize_t
    tol::Float64
    max_subspace::Csize_t   # RPA thick-restart bound; 0 = the kernel's auto bound
    readout::Cint   # SECTOR_READOUTS
    metric::Cint    # RPA_METRIC_CODES
    kernel::Cint    # RPA_KERNEL_CODES
    reserved::Cint  # explicit C padding; stays 0
end

"""The knobs of the same-sector Tamm–Dancoff readout; see [`tda`](@ref). Mirrors the library's
`TdaDavidsonOpts` one-to-one; `k` and `seed` are call parameters, not fields. `tol` is an ABSOLUTE
residual tolerance per root, in hartree (finite, `≥ 0`; default `1e-8`); `max_iter` counts outer
block-Davidson iterations (default 300); `max_subspace` is the thick-restart bound (`0` = the auto
`max(8(k + 2), 32)`); `restart_keep` the Ritz columns kept across a restart (`0` = `2k`); `nwarm`
the unpreconditioned warm-up iterations; `max_concurrent` the most `A` images computed at once
(`0` = a whole batch; it bounds memory only, the values do not depend on it). `precond` is a wire
code — pass it as the Symbol [`QNTT.TDA_PRECOND_CODES`](@ref) names (default `:auto`, which runs as
`:diag` at `κ = 10`); `precond_kappa` caps `:diag`'s condition number (`≤ 0` or non-finite = the
auto cap `1e3`; default `10`, the value `:auto` resolves to) and the other arms ignore it.
`verbosity ≥ 1` prints the per-iteration line to stdout."""
struct TdaOptions
    tol::Float64
    max_iter::Csize_t
    max_subspace::Csize_t
    restart_keep::Csize_t
    nwarm::Csize_t
    max_concurrent::Csize_t
    precond_kappa::Float64
    precond::Cint       # TDA_PRECOND_CODES
    verbosity::Cint
end

"""The exposed knobs of `qntt_f64_sector_alpha`; see [`sector_alpha`](@ref). Mirrors the
library's own `SectorStaticOpts` one-to-one, so no field is inert on the entry point that receives
it — [`SectorMomentOptions`](@ref) is the other half. `sym_assert_rel` is honoured only when
`has_sym_assert_rel != 0`; leave it at the default and the library recomputes `100·tol` from the
`tol` you actually passed, which is what keeps a healthy loosened-tol solve from being blamed for
a broken assembly. `route` is a wire code — pass it as the Symbol [`QNTT.PROP_ROUTES`](@ref)
names."""
struct SectorAlphaOptions
    tol::Float64
    max_iter::Csize_t
    tt_grad_tol::Float64
    sym_assert_rel::Float64
    has_sym_assert_rel::Cint
    route::Cint     # PROP_ROUTES
end

"""The exposed knobs of `qntt_f64_sector_transition_moments`; see
[`sector_transition_moments`](@ref). Mirrors the library's own `SectorMomentOpts` minus `solver`,
which only the iterative route reads and which the C ABI does not expose. `degeneracy_tol` is not
a tolerance on the answer but a statement about which numbers EXIST: inside a degenerate cluster
the individual `T_n` are an arbitrary orthogonal mixing. `reserved` removes the tail padding and
must stay `0`."""
struct SectorMomentOptions
    tt_grad_tol::Float64
    degeneracy_tol::Float64
    psd_tol::Float64
    route::Cint     # PROP_ROUTES
    reserved::Cint
end

"""Which route builds a channel's property covector, keyed by keyword. `:materialized` is the only
route a shifted channel HAS: `:shifted` needs `μ̂` as an operator on `x`'s own space, which a
cross-irrep property is not, and passing it raises [`QNTTError`](@ref) rather than silently
aliasing. Present so ONE vocabulary crosses the abelian and SU(2) layers."""
const PROP_ROUTES = Dict(:shifted => Cint(0), :materialized => Cint(1))

"""Which PHYSICS refusal a cross-sector observable returned, keyed by wire code — the `refusal`
field of the [`QNTTError`](@ref) [`sector_alpha`](@ref) and [`sector_transition_moments`](@ref)
throw, so a caller branches on the code instead of parsing the message. `:none` is what a CALLER
error carries: a bad argument is a statement about the arguments, not about the state.

`:not_tt_critical` — `B^φ` contracts against the normal residual and is invalid off criticality;
`:hessian_indefinite` — `H^rr` is indefinite on the channel, i.e. `x` is not the lowest state of
`Γ0 ⊕ φ` (physics, not failure); `:symplectic_unstable` — both Hessian blocks are indefinite, so
no real symplectic basis exists at all; `:solve_not_converged` — a Krylov solve hit its cap, and a
partial CG answer is a lower bound of unknown tightness; `:uncertified_negative` — the iterative
moment route's, which these bindings do not reach. `:negative_omega`, `:near_resonance` and
`:indefinite_above_pole` (7–9) belong to the frequency-dependent `α(ω)`, which stays Rust-only:
they are mapped so that entry point's refusals are discriminated the day it crosses, and nothing
writes them today. Code `6` is deliberately absent — a multi-class flux is a caller error and
arrives as status 2 with `:none`."""
const SECTOR_REFUSALS = Dict(Cint(0) => :none,
                             Cint(1) => :not_tt_critical,
                             Cint(2) => :hessian_indefinite,
                             Cint(3) => :symplectic_unstable,
                             Cint(4) => :solve_not_converged,
                             Cint(5) => :uncertified_negative,
                             Cint(7) => :negative_omega,
                             Cint(8) => :near_resonance,
                             Cint(9) => :indefinite_above_pole)

# The scalars of `sector_alpha`, as the C ABI writes them. Internal: the high level unpacks it
# into the named tuple that entry point returns.
struct SectorAlphaInfo
    alpha_defect::Float64
    sigma_min::Float64
    dim::Csize_t
    phi::Cint
    target::Cint
end

# The scalars and SHAPE of `sector_transition_moments`, as the C ABI writes them. `n_modes`,
# `n_neg` and `n_clusters` are the three lengths a caller cannot predict; all three are bounded by
# `dim`. Internal, like `SectorAlphaInfo`.
struct SectorMomentsInfo
    orth_error::Float64
    sigma_min::Float64
    dim::Csize_t
    n_modes::Csize_t
    n_neg::Csize_t
    n_clusters::Csize_t
    metric::Cint
    has_alpha_sos::Cint
    complete_basis::Cint
    phi::Cint
    target::Cint
    reserved::Cint
end

"""`core | active | virtual` split of the orbital chain, in site order; see
[`cas_reference_state`](@ref). Built from a 3-tuple `(n_core, n_active, n_virt)`."""
struct CasPartition
    n_core::Csize_t
    n_active::Csize_t
    n_virt::Csize_t
end
CasPartition(cas::NTuple{3,Integer}) = CasPartition(cas[1], cas[2], cas[3])
CasPartition(cas::CasPartition) = cas

"""The exposed subset of the joint DMRG-SCF driver's options; see [`casscf`](@ref)."""
struct CasscfOptions
    orb_max_iter::Csize_t
    orb_grad::Float64
    tt_grad_tol::Float64
    orb_floor::Float64
    tt_preserve_grid::Cint   # nonzero (the default HERE) keeps the pinned grid through TT legs
    construction::Cint       # 0 = layer default, else CONSTRUCTIONS; ignored on SU(2)
    verbose::Cint
end

"""Flat diagnostics of a joint run (the vectors ride a separate handle); see [`casscf`](@ref)."""
struct CasscfInfo
    converged::Cint
    rank_limited::Cint
    stalled_at_floor::Cint
    iters::Csize_t
    n_eff::Csize_t
    ls_build_count::Csize_t
    tt_legs::Csize_t
    rejections::Csize_t
    sigma_min::Float64
    u_orth_dev::Float64
    g_dropped_final::Float64
end

"""Gentlest-ascent dynamics options; see [`gad`](@ref)."""
struct GadOptions
    gradtol::Float64
    max_steps::Csize_t
    alpha0::Float64
    alpha_min::Float64
    alpha_max::Float64
    k_davidson::Csize_t
    w_model::Cint            # 0 = switch (default), 1 = damped, 2 = full
    w_damp_s::Float64        # damped only: > 0 fixes s, <= 0 selects the gradient-tracked rule
    b_switch_tol::Float64    # switch only
    floor_kind::Cint         # 0 = default hard floor, 1 = hard, 2 = tikhonov
    floor::Float64
    certify::Cint
    probe_budget::Csize_t
    schmidt_floor::Float64
    seed::UInt64
    preserve_grid::Cint      # REQUIRED on a sector-pinned (CAS) grid
    rank_adapt::Cint         # 0 = fixed grid (default), 1 = both (grow and shrink), 2 = shrink only
    chi::Csize_t             # both only: pooled total bond-dimension cap of a growth round
                             # (SU(2): reduced multiplets), 0 = uncapped
    rank_eps::Float64        # adaptation energy budget (relative); positive, finite
    rank_period::Csize_t     # adapt every rank_period accepted steps (>= 1)
    border_gain_max::Float64 # manifold border: no certificate above this Weingarten gain (1.0)
end

"""Flat GAD diagnostics; the optional fields carry a `has_*` flag. See [`gad`](@ref)."""
struct GadInfo
    converged::Cint
    index1_certified::Cint
    has_index_estimate::Cint
    index_estimate::Csize_t
    steps::Csize_t
    applies::Csize_t
    gnorm::Float64
    lambda::Float64
    has_eig1::Cint
    eig1::Float64
    v_overlap::Float64
    b_gain_max::Float64
    min_schmidt_min::Float64
    schmidt_floor_hit::Cint
    rejected_steps::Csize_t
    relatches::Csize_t
    regime_switches::Csize_t
    radial_overlap_max::Float64
    branch_gap::Float64
    has_witness_overlap::Cint
    witness_overlap::Float64
    tracked_mode_is_lowest::Cint
    grid_changes::Csize_t
    border_verdict::Cint
    b_gain_end::Float64
end

"""State-average MALS options; the MALS block is nested and the weight / rank / noise schedules
travel separately. See [`state_average_mals`](@ref)."""
struct StateAverageOptions
    mals::MalsOptions
    homing::Cint             # 0 (default) labels roots by energy at every bond; 1 follows them
    reorthonormalize::Cint   # 1 (default): Löwdin-orthonormalize the truncated centres
    scale::Float64           # relative size of the start perturbation of centres 2..n
    maxsubspace::Csize_t     # 0 = the driver's own max(DAVIDSON_MAXSUBSPACE, 8n)
    restart_keep::Csize_t    # 0 = the driver's own 2n, clamped below maxsubspace
end

"""One state-average outer iteration, as returned by [`state_average_mals`](@ref) in its
`history`. The per-root energies and discarded weights are not exposed here — the pooled
`discarded_weight` is what the shared truncation actually chopped."""
struct SaSweep
    energy::Float64            # Σᵢ wᵢ λᵢ — what the sweep minimizes
    max_dev::Float64           # maxᵢ |Δλᵢ| / |λᵢ| over the roots
    overlap_dev::Float64       # drift from orthogonality before the Löwdin step
    discarded_weight::Float64  # POOLED over the roots
    max_bond_dim::Csize_t      # largest pooled total bond dimension reached in this sweep
    bond_dim::Csize_t          # the pooled cap in force this iteration (0 = uncapped)
    noise::Float64
    numops::Csize_t
end

"""State-specific MALS options; see [`state_specific_mals`](@ref)."""
struct StateSpecificOptions
    mals::MalsOptions        # unlike the state average, this path accepts `noise_type = :random`
    mode::Cint               # 0 = penalty (the default), 1 = project
    penalty_rule::Cint       # 0 = auto (the default), 1 = fixed
    penalty_value::Float64   # read under `penalty_rule == 1` only
    roots::Csize_t           # local roots per bond; > 1 homes onto the incoming block
    maxsubspace::Csize_t     # 0 = the driver's own default
    restart_keep::Csize_t    # 0 = the driver's own default
end

"""One state-specific outer iteration, as returned by [`state_specific_mals`](@ref) in its
`history`."""
struct SsSweep
    energy::Float64            # the PLAIN Rayleigh quotient of H alone — the reported energy
    penalized::Float64         # the penalized Ritz value the sweep actually minimizes
    discarded_weight::Float64
    max_bond_dim::Csize_t      # largest pooled total bond dimension reached in this sweep
    bond_dim::Csize_t          # the pooled cap in force this iteration (0 = uncapped)
    noise::Float64
    penalty::Float64           # the weight in force; 0 in project mode and with no lower states
    swaps::Csize_t
    numops::Csize_t            # 0 with no lower states and `roots = 1` (that path delegates to `mals`)
    skipped::Csize_t
    converged::Cint
end

"""Excited-state ladder options; the four schedules travel separately and an empty one means the
driver's own value. `bond_dims` (of the four schedules) is a pooled total bond dimension schedule
(SU(2): reduced multiplets), like `rank`. See [`excited_states`](@ref)."""
struct ExcitedOptions
    nroots::Csize_t
    rank::Csize_t            # pooled total bond dimension (SU(2): reduced multiplets) cap of the
                             # sweeps and of the polish manifold; 0 = full rank
    eps::Float64
    max_iter::Csize_t
    noise_type::Cint         # 0 = random (refused here), 1 = perturbative (the default)
    seed::UInt64
    mode::Cint               # 0 = penalty, 1 = project
    penalty_rule::Cint       # 0 = auto, 1 = fixed
    penalty_value::Float64   # also fixes the POLISH weight
    ss_start::Cint           # 0 = each rung starts from its state-average state (1 is reserved)
    polish::Cint             # 0 returns the sweep's own state; the gradnorm is then NaN
    lower_grad_tol::Float64
    trim_tol::Float64
    trim_passes::Csize_t
    cg_eps::Float64          # 1e-13: the pipeline override, not `RiemannianOptions`' own 1e-4
    cg_max_iter::Csize_t
    cg_gradtol::Float64      # inert here; carried for API stability
    cg_seed::UInt64
    cg_variant::Cint         # 0 = Fletcher-Reeves, 1 = Polak-Ribière
    cg_restarts::Csize_t
    polish_tol::Float64
    grad_gate::Float64
    overlap_tol::Float64
    keep_sa::Cint            # nonzero also returns the state-average ladder
    verbose::Cint
end

"""The ladder's scalar diagnostics; the per-root rows are [`ExcitedRoot`](@ref). See
[`excited_states`](@ref)."""
struct ExcitedInfo
    overlap_dev::Float64     # the largest off-diagonal entry of the returned Gram
    swaps::Csize_t
    enriched::Cint           # the enrichment iteration ran (a rank-1 start)
    licence::Cint            # convergence AND orthogonality AND ordering — NEVER completeness
end

"""One row of the ladder's per-root record; see [`excited_states`](@ref)."""
struct ExcitedRoot
    sa_energy::Float64       # the state-average stage's energy for this root
    ss_energy::Float64       # the state-specific sweep's own energy, BEFORE the polish
    j_ss::Float64
    j_pol::Float64
    gradnorm::Float64        # the exact penalized gate; NaN when `polish = false`
    penalty::Float64         # 0 when this root had no lower states
    passes::Csize_t
    stalled::Cint
    converged::Cint
    swapped::Cint
end

for (S, fn) in ((:AlsOptions, :qntt_als_options_default),
                (:MalsOptions, :qntt_mals_options_default),
                (:TdvpOptions, :qntt_tdvp_options_default),
                (:LanczosOptions, :qntt_lanczos_options_default),
                (:SymplecticOptions, :qntt_symplectic_options_default),
                (:SectorOptions, :qntt_sector_options_default),
                (:TdaOptions, :qntt_tda_options_default),
                (:SectorAlphaOptions, :qntt_sector_alpha_options_default),
                (:SectorMomentOptions, :qntt_sector_moment_options_default),
                (:CasscfOptions, :qntt_casscf_options_default),
                (:GadOptions, :qntt_gad_options_default),
                (:StateAverageOptions, :qntt_state_average_options_default),
                (:StateSpecificOptions, :qntt_state_specific_options_default),
                (:ExcitedOptions, :qntt_excited_options_default))
    @eval function defaults(::Type{$S})
        out = Ref{$S}()
        check(ccall(($(QuoteNode(fn)), libqntt), Cint, (Ptr{$S},), out))
        return out[]
    end
end

function defaults(::Type{RiemannianOptions})
    out = Ref{RiemannianOptions}()
    check(ccall((:qntt_riemannian_options_default, libqntt), Cint,
                (Ptr{RiemannianOptions},), out))
    return out[]
end

"""
    QNTT.setfields(opts; kwargs...)

Copy an options struct with some fields replaced. Booleans are converted to the C flags and
`Symbol`s to the documented enum codes, so `verbose = true` and `variant = :pr` both work.
"""
function setfields(opts::S; kwargs...) where {S}
    vals = Any[getfield(opts, f) for f in fieldnames(S)]
    for (k, v) in kwargs
        i = findfirst(==(k), fieldnames(S))
        i === nothing && throw(ArgumentError("$S has no field $k"))
        vals[i] = _coerce(fieldtype(S, k), k, v)
    end
    return S(vals...)
end

_coerce(::Type{T}, ::Symbol, v) where {T} = convert(T, v)
_coerce(::Type{Cint}, ::Symbol, v::Bool) = Cint(v)
_coerce(::Type{Cint}, name::Symbol, v::Symbol) = Cint(_enum_code(name, v))

"""Which quantity [`sector_rpa`](@ref) reports, keyed by keyword: `:tda` = `eig(A^φ)`, an
ENERGY; `:rpa` = the signed `ω²` of the pair `(A^φ, B^φ)`. One readout per entry point in the
library, so this selects the arm; the two are different quantities in different units and are
**not** interchangeable."""
const SECTOR_READOUTS = Dict(:tda => Cint(0), :rpa => Cint(1))

"""Which Hessian block carries the W-metric, keyed by keyword: `:ii` (`W = H^ii`), `:rr` (the
swap), `:auto` — `:ii` everywhere, with no probe: on [`rpa_signed`](@ref) and [`sector_rpa`](@ref)
the side `q = X ∓ Y` the Weingarten fraction reads and, under `kernel = :signed`, also the signed
kernel's lane, with no retry (neither kernel refuses on the metric). The INVERSE of
`QNTT.RPA_METRICS`, which maps the same codes back to keywords — and **not** `QNTT.HESS_PARTS`,
which numbers `:rr` first. The asymmetry is frozen: it is what the C `out_metric` has shipped since
metric selection landed."""
const RPA_METRIC_CODES = Dict(:ii => Cint(0), :rr => Cint(1), :auto => Cint(2))

"""Curvature regularization of [`rpa_signed`](@ref) and [`sector_rpa`](@ref), keyed by keyword —
the C `QNTT_CURVATURE_REG_*` codes. `:off` is the exact Riemannian Hessian; `:hard` replaces the
curvature term `B` by `P·B·P`, `P` the column projector off the right Schmidt directions with
Schmidt value below `curvature_scale`; `:tikhonov` by `D·B·D`, `D = V diag(s²/(s² + ε²)) Vᵀ`,
`ε = curvature_scale`. The scale is an absolute Schmidt value — on the SU(2) layer the RAW `σ` of
`QNTT.schmidt` (not `√weight·σ`), so at a singlet one scale cuts the same directions on both
layers. `A` and the tangent space are untouched; on a shifted channel
the same projector acts on the relabelled column labels (the channel's block of the untyped
regularized operator)."""
const CURVATURE_REG_CODES = Dict(:off => Cint(0), :hard => Cint(1), :tikhonov => Cint(2))

"""The preconditioner of [`sector_rpa`](@ref), keyed by keyword — the C `QNTT_SECTOR_PRECOND_*`
codes. `:off` is none; `:d0` the channel's fixed diagonal inverse (of `D_A²` on the `:rpa` readout,
of `D_A` on `:tda`, `D_A` the shifted effective-Hamiltonian diagonal minus `λ`), its dynamic range
capped by `precond_kappa` (`≤ 0` or non-finite = the auto cap `1e3`); `:auto` the library's sector
default, which is `:off` until a measured table promotes `:d0`."""
const SECTOR_PRECOND_CODES = Dict(:off => Cint(0), :d0 => Cint(1), :auto => Cint(2))

"""The preconditioner of [`tda`](@ref), keyed by keyword — the C `QNTT_TDA_PRECOND_*` codes.
`:auto` (the default) runs as `:diag` at `κ = 10`; `:off` is plain residual corrections; `:diag` is
D0, the fixed θ-independent inverse of the clamped diagonal `D_A` of `A` (the effective-Hamiltonian
diagonal minus `λ`), its condition number capped by `precond_kappa`; `:shifted` the classical
`(D_A − θᵢ)⁻¹` correction at each root's Ritz value. **Not** the numbering of
[`QNTT.SECTOR_PRECOND_CODES`](@ref) (which puts `:auto` last): a different enum over a different
operator. The inverse map is `QNTT.TDA_PRECONDS`."""
const TDA_PRECOND_CODES = Dict(:auto => Cint(0), :off => Cint(1), :diag => Cint(2),
                               :shifted => Cint(3))

"""The preconditioner a [`tda`](@ref) solve RAN, keyed by wire code (`info.precond`): `:auto`
resolved (to `:diag`), and `:off` where the `:diag` guard skipped a collapsed diagonal model. The
inverse of [`QNTT.TDA_PRECOND_CODES`](@ref)."""
const TDA_PRECONDS = Dict(v => k for (k, v) in TDA_PRECOND_CODES)

"""The TRI-state [`sector_rpa_dense`](@ref) reads out of the C `out_stable`, keyed by wire code.
`:stable` = both Hessian blocks PD, every `ω² > 0`; `:resonant` = exactly one block PD, the signed
`ω²` is real and a negative entry is a resonance `ω = i√|ω²|` — physics, not a failure;
`:unstable` = both blocks indefinite, no real `ω²` exists at all. Julia never RETURNS the third: it
is the one arm the C call reports as a status, so [`sector_rpa_dense`](@ref) throws
[`QNTTError`](@ref) there. A caller testing `status !== :stable` must handle `:resonant` as well."""
const SECTOR_STATUS = Dict(Cint(1) => :stable, Cint(0) => :resonant, Cint(-1) => :unstable)

const _ENUMS = Dict(
    :variant => Dict(:fr => 0, :pr => 1),
    :noise_type => Dict(:random => 0, :perturbative => 1),
    :construction => Dict(:default => 0, :median => 1, :bipartite => 2,
                          :bipartite_per_sector => 3),
    :w_model => Dict(:switch => 0, :damped => 1, :full => 2),
    :floor_kind => Dict(:default => 0, :hard => 1, :tikhonov => 2),
    :mode => Dict(:penalty => 0, :project => 1),
    :penalty_rule => Dict(:auto => 0, :fixed => 1),
    :ss_start => Dict(:sa => 0, :x0 => 1),
    :cg_variant => Dict(:fr => 0, :pr => 1),
    :rank_adapt => Dict(:fixed => 0, :both => 1, :shrink => 2),
    # Derived from the exported tables rather than retyped, so the two cannot drift.
    :readout => Dict(k => Int(v) for (k, v) in SECTOR_READOUTS),
    :metric => Dict(k => Int(v) for (k, v) in RPA_METRIC_CODES),
    :route => Dict(k => Int(v) for (k, v) in PROP_ROUTES),
    # `TdaOptions.precond` — the only options field named `precond`.
    :precond => Dict(k => Int(v) for (k, v) in TDA_PRECOND_CODES),
)

function _enum_code(name::Symbol, v::Symbol)
    tbl = get(_ENUMS, name, nothing)
    tbl === nothing && throw(ArgumentError("field $name does not take a Symbol"))
    haskey(tbl, v) || throw(ArgumentError("$name must be one of $(keys(tbl)), got :$v"))
    return tbl[v]
end

# `_penalty_fields(penalty) -> (rule, value)`: the `(penalty_rule, penalty_value)` pair behind the
# single `penalty` keyword of `state_specific_mals` and `excited_states`. `:auto` is the measured
# rule `w = max(10(λ⊥ − minⱼEⱼ), 1)` with its collapse escalation; a number is used verbatim and is
# UNGUARDED — below the gap to the target, the lower state is its own minimizer.
function _penalty_fields(p::Symbol)
    p === :auto && return (Cint(0), 0.0)
    throw(ArgumentError("penalty must be :auto or a non-negative Real, got :$p"))
end
function _penalty_fields(w::Real)
    (isfinite(w) && w >= 0) ||
        throw(ArgumentError("penalty must be finite and non-negative, got $w"))
    return (Cint(1), Float64(w))
end

# `_curvature_fields(curvature, curvature_scale) -> (kind, value)`: the `(curvature_kind,
# curvature_value)` pair behind the `curvature` / `curvature_scale` keywords of `rpa_signed` and
# `sector_rpa` (`CURVATURE_REG_CODES`). A caller error is an `ArgumentError` BEFORE the `ccall`: an
# unknown Symbol, or `:hard` / `:tikhonov` with a scale that is not finite and `> 0`. `:off` ignores
# its scale (NaN included, the C contract). An integer code passes through unchecked, and the C
# side refuses a bad one (`QNTTError`).
function _curvature_fields(c::Symbol, scale::Real)
    haskey(CURVATURE_REG_CODES, c) ||
        throw(ArgumentError("curvature must be one of $(sort(collect(keys(CURVATURE_REG_CODES)))), got :$c"))
    c === :off && return (CURVATURE_REG_CODES[c], Float64(scale))
    (isfinite(scale) && scale > 0) ||
        throw(ArgumentError("curvature = :$c needs a finite curvature_scale > 0 (an absolute Schmidt value), got $scale"))
    return (CURVATURE_REG_CODES[c], Float64(scale))
end
_curvature_fields(c::Integer, scale::Real) = (Cint(c), Float64(scale))

# `_precond_fields(precond, precond_kappa) -> (code, kappa)`: the `(precond, precond_kappa)` pair
# behind the keywords of `sector_rpa` (`SECTOR_PRECOND_CODES`). An unknown Symbol is an
# `ArgumentError` BEFORE the `ccall`; `precond_kappa` is never refused (a value `≤ 0` or non-finite
# is the auto cap). An integer code passes through unchecked, and the C side refuses a bad one
# (`QNTTError`).
function _precond_fields(p::Symbol, kappa::Real)
    haskey(SECTOR_PRECOND_CODES, p) ||
        throw(ArgumentError("precond must be one of $(sort(collect(keys(SECTOR_PRECOND_CODES)))), got :$p"))
    return (SECTOR_PRECOND_CODES[p], Float64(kappa))
end
_precond_fields(p::Integer, kappa::Real) = (Cint(p), Float64(kappa))

# Wire codes shared with `capi/src/enums.rs`. Kept as small lookup tables rather than Julia enums
# so the numbers stay visible next to the names.
const ROUND_MODES = Dict(:abs => Cint(0), :rel => Cint(1))
const CONSTRUCTIONS = Dict(:default => Cint(0), :median => Cint(1),
                           :bipartite => Cint(2), :bipartite_per_sector => Cint(3))
const TWO_BODY_SYMMETRIES = Dict(:none => Cint(0), :pairswap => Cint(1), :eightfold => Cint(2))
const THREE_BODY_SYMMETRIES = Dict(:none => Cint(0), :pairperm => Cint(1), :full48 => Cint(2))
const FAMILIES = Dict(:empty => Cint(0), :up => Cint(1), :dn => Cint(2), :occ => Cint(3))

"""Which Hessian part [`pd_probe`](@ref) targets. `:rr` is `A + B`, the real-layer energy Hessian
(indefinite ⇒ the point is a saddle of the fixed-rank energy); `:ii` is `A − B`, the curvature
along the imaginary directions (indefinite ⇒ a physical instability). On a `ComplexF64` state
there is only one object — the complex Riemannian Hessian `𝔥` — reached with `:rr`; `:ii` is
rejected there, because `−J𝔥J ≻ 0 ⟺ 𝔥 ≻ 0` and that single probe certifies the whole RPA
precondition."""
const HESS_PARTS = Dict(:rr => Cint(0), :ii => Cint(1))

"""PD-probe verdicts, keyed by wire code. `:witness_negative` is a PROOF — a fresh-apply Rayleigh
quotient below zero, valid however unconverged the Davidson was. `:likely_pd` is evidence with a
margin, and only means something read next to its `residual`. `:inconclusive` means "no negative
direction found in this budget" and is **not** a positive-definiteness claim."""
const PD_VERDICTS = Dict(Cint(0) => :witness_negative, Cint(1) => :likely_pd,
                         Cint(2) => :inconclusive)

# Abelian point-group kinds. The codes are the Rust `PointGroupKind` declaration order, which also
# indexes the crate's Molpro↔XOR and name tables — frozen, never renumbered.
const POINT_GROUPS = Dict(:C1 => Cint(0), :Ci => Cint(1), :C2 => Cint(2), :Cs => Cint(3),
                          :D2 => Cint(4), :C2v => Cint(5), :C2h => Cint(6), :D2h => Cint(7))
const POINT_GROUP_NAMES = Dict(v => k for (k, v) in POINT_GROUPS)
"""Sentinel accepted only by [`read_fcidump`](@ref)'s `point_group = :auto`: guess the kind from
`max(ORBSYM)`. The guess affects irrep NAMES only — all same-order abelian groups share one
product table, so no energy or selection rule depends on it."""
const PG_AUTO = Cint(-1)

function _code(tbl::Dict{Symbol,Cint}, v::Symbol, what::AbstractString)
    haskey(tbl, v) || throw(ArgumentError("$what must be one of $(sort(collect(keys(tbl)))), got :$v"))
    return tbl[v]
end
_code(::Dict{Symbol,Cint}, v::Integer, ::AbstractString) = Cint(v)

"""Joint-RPA statuses, keyed by wire code; see [`joint_rpa`](@ref). None throws: they are
diagnoses about the state. `:refused` is the joint context's own criticality / PD gate."""
const JOINT_RPA_STATUS = Dict(Cint(0) => :stable, Cint(1) => :resonant,
                              Cint(2) => :unstable, Cint(3) => :refused)

# Per-iteration series of a joint run, `qntt_casscf_diag_series` codes.
const _CASSCF_SERIES = (e_history = Cint(0), g_orb_kept = Cint(1), g_orb_full = Cint(2),
                        g_orb_dropped = Cint(3), tt_gradnorms = Cint(4), orb_spectrum = Cint(5),
                        tr_radii = Cint(6), rho_history = Cint(7))

"""Which Hessian block served as the metric of [`rpa_signed`](@ref) or [`sector_rpa`](@ref),
keyed by wire code. `:auto` is the UNRESOLVED route: the run returned before choosing a metric
(a `sector_rpa` channel with nothing to solve, or whose start block the projector annihilated), or
the readout has no metric at all (the `:tda` arm). Anywhere `info.applies > 0` on the `:rpa` arm
the code is `:ii` or `:rr` and names the block the solve ran on. The inverse map is
`QNTT.RPA_METRIC_CODES`."""
const RPA_METRICS = Dict(Cint(0) => :ii, Cint(1) => :rr, Cint(2) => :auto)

"""The kernel behind [`rpa_signed`](@ref), keyed by wire code: `:paired` (the library default, the
paired Davidson) or `:signed` (the signed W-metric Davidson, run when a solver option has no paired
arm)."""
const RPA_KERNELS = Dict(Cint(0) => :paired, Cint(1) => :signed)

"""The `kernel` request of [`rpa_signed`](@ref) and [`sector_rpa`](@ref), keyed by keyword — the C
`QNTT_RPA_KERNEL_*` codes:
`:paired` (the library default, the paired Davidson) or `:signed` (the signed W-metric Davidson,
on the `Float64` layers on the lane `metric` names, on the complex layers always `:rr`). The
inverse of `QNTT.RPA_KERNELS`."""
const RPA_KERNEL_CODES = Dict(:paired => Cint(0), :signed => Cint(1))
# `SectorOptions.kernel` takes the Symbol through `setfields` too (derived, never retyped).
_ENUMS[:kernel] = Dict(k => Int(v) for (k, v) in RPA_KERNEL_CODES)

# The 5-entry report of the RPA exports as a NamedTuple; a `-1` (the kernel that ran has no such
# number) reads `nothing`.
function _rpa_report(r::AbstractVector{<:Integer})
    v(i) = r[i] < 0 ? nothing : Int(r[i])
    return (; neg_apb = v(1), neg_amb = v(2), metric_events = v(3), n_negative_basis = v(4),
            general_path = r[5] < 0 ? nothing : r[5] != 0)
end
