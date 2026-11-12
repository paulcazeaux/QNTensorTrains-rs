# Copyright (C) 2023-2026 Paul Cazeaux
#
# This file is part of qntt. qntt is free software: you can
# redistribute it and/or modify it under the terms of the GNU
# General Public License as published by the Free Software
# Foundation, either version 3 of the License, or (at your
# option) any later version. See LICENSE for details.

# =================================================================================================
# Solvers
# =================================================================================================

"""
    als(H, x0; eps = 1e-4, max_iter = 20, verbose = false) -> (λ, x)

One-site alternating least squares (DMRG). The rank grid of `x0` is preserved — ALS optimizes on
a fixed manifold; use [`mals`](@ref) to grow ranks.
"""
function als(h::Hamiltonian{T,L}, x0::TTVector{T,L}; kwargs...) where {T,L}
    opts = setfields(defaults(AlsOptions); kwargs...)
    λ, ptr = _als(L, T, _ptr(h), _ptr(x0), opts)
    return (λ, TTVector{T,L}(ptr))
end

"""
    mals(H, x0; bond_dims = [0], noise = Float64[], noise_type = :perturbative, eps = 1e-6,
         conv_tol = 0.0, max_iter = 20, seed = 1234, verbose = false) -> (λ, x, history)

Two-site rank-adaptive DMRG.

`bond_dims` is the per-outer-iteration cap on the **total** bond dimension summed over symmetry
sectors — the conventional DMRG `M`. `0` means uncapped, and the last entry is held once the sweep
count exceeds the schedule.

`noise` is the matching per-iteration noise amplitude. Noise lets a symmetry sector with no weight
in the current state survive truncation, which is what breaks the fixed point a single determinant
otherwise sits at — started there, a noise-free sweep converges, looks converged, and stops far
above the minimum.

!!! warning "End the schedule at zero"
    The driver will not stop on a noisy iteration under either flavour, so every schedule must end
    at `0.0`. A typical shape is a decaying ramp: `noise = [1e-4, 1e-4, 1e-5, 1e-5, 0.0]`. Under
    `:random` the noise-free tail is also the only part that carries a variational energy at all
    (that flavour perturbs the state, not merely the choice of kept basis); `:perturbative` is
    variational throughout, and the tail is what settles it back onto the noise-free answer.

`noise_type` selects the noise flavour. **`:perturbative` is the default** — White
density-matrix noise: variational throughout, deterministic, exact at full rank, and measured to
track/beat random on N₂/cc-pVDZ Fiedler. Each operator channel is unit-trace normalized and `α` is
globally rescaled by `‖ρ_noise‖_F`, so a single nominal amplitude (~1e-4) is portable across bonds
and molecules. `:random` (isotropic perturbation of the two-site block) is kept as the backup, for
A/B comparison and for the one case where it is preferred on the merits: from a **bare** determinant
or CSF start the perturbative enrichment is built from the current state and cannot leave its
subspace, so the sweep stalls at a genuine critical point above the true ground — either fill the
start's rank grid with random content first, or run that leg on `:random`. Both flavours work on the
abelian and the SU(2) layer — on SU(2) the perturbative enrichment is built in the θ-scaled metric.

Note that the two flavours read `noise` differently: on `:perturbative` an entry is a *nominal*
amplitude, rescaled to `α/‖ρ_noise‖_F` before it enters `ρ' = ρ + α_eff·ρ_noise`; on `:random` it is
the relative Frobenius amplitude of the perturbation itself. `~1e-4` is the right starting order
either way.

`eps` is otherwise three knobs at once — the per-bond truncation budget, the inner Davidson
floor and the stopping test on the relative energy improvement. `conv_tol > 0` takes over the
stopping test alone, so `eps = 1e-2, conv_tol = 1e-10` is a loose truncation swept to convergence
(what a rank-adaptive stage is meant to be); `conv_tol = 0` (the default) stops on `eps`, exactly
the historical behaviour.

`history` is a vector of [`MalsSweep`](@ref) — per-iteration energy, discarded weight, bond
dimension and noise. The discarded weight is what tells you whether a rank ramp has converged.

```julia
λ, x, hist = QNTT.mals(H, hf; bond_dims = [8, 16, 32, 64],
                       noise = [1e-2, 1e-3, 1e-4, 0.0], eps = 1e-9, max_iter = 20)
last(hist).discarded_weight
```
"""
function mals(h::Hamiltonian{T,L}, x0::TTVector{T,L};
              bond_dims = [0], noise = Float64[], kwargs...) where {T,L}
    opts = setfields(defaults(MalsOptions); kwargs...)
    λ, ptr, hist = _mals(L, T, _ptr(h), _ptr(x0), opts, bond_dims, noise)
    return (λ, TTVector{T,L}(ptr), hist)
end

"""
    tdvp(H, x0; dtau = 0.1, eps = 1e-4, trunc = 1e-8, max_sweeps = 100, chi = 0,
         exptol = 1e-10, verbose = false) -> (λ, x)

Imaginary-time two-site TDVP ground-state search. `chi` is a pooled total bond-dimension cap
(SU(2): reduced multiplets); `0` = none.
"""
function tdvp(h::Hamiltonian{T,L}, x0::TTVector{T,L}; kwargs...) where {T,L}
    opts = setfields(defaults(TdvpOptions); kwargs...)
    λ, ptr = _tdvp(L, T, _ptr(h), _ptr(x0), opts)
    return (λ, TTVector{T,L}(ptr))
end

"""
    lanczos(H, x0; tol = 1e-4, max_iter = 20) -> (λ, x, info)

Rounded Lanczos. `info` is a named tuple with the per-iteration history
(`lambdas`, `res`, `ranks`, `rounded_ranks`) and a `breakdown` flag.
"""
function lanczos(h::Hamiltonian{T,L}, x0::TTVector{T,L}; kwargs...) where {T,L}
    opts = setfields(defaults(LanczosOptions); kwargs...)
    λ, ptr, brk, lambdas, res, rk, rrk = _lanczos(L, T, _ptr(h), _ptr(x0), opts)
    info = (breakdown = brk, lambdas = lambdas, res = res,
            ranks = rk, rounded_ranks = rrk)
    return (λ, TTVector{T,L}(ptr), info)
end

"""
    riemannian(H, x0; cg = true, eps = 1e-4, max_iter = 100, gradtol = 1e-6, seed = 1234,
               variant = :pr) -> (λ, x)

Riemannian optimization on the fixed-rank TT manifold: conjugate gradients (`cg = true`) or
steepest descent.
"""
function riemannian(h::Hamiltonian{T,L}, x0::TTVector{T,L};
                    cg::Bool = true,
                    kwargs...) where {T,L}
    opts = setfields(defaults(RiemannianOptions); kwargs...)
    λ, ptr = _riemannian(L, T, _ptr(h), _ptr(x0), opts, Cint(cg))
    return (λ, TTVector{T,L}(ptr))
end

# =================================================================================================
# Excited states
# =================================================================================================

# The state-average and state-specific option structs nest a `MalsOptions`, so the sweep keywords
# (`eps`, `max_iter`, `seed`, `noise_type`, `conv_tol`, `verbose`) have to be routed into it while
# the multi-state keywords stay on the outer struct.
function _split_mals_kwargs(kwargs)
    inner, outer = Pair{Symbol,Any}[], Pair{Symbol,Any}[]
    for (k, v) in kwargs
        push!(k in fieldnames(MalsOptions) ? inner : outer, k => v)
    end
    return (inner, outer)
end

function _lower_ptrs(lower, ::Type{T}, ::Type{L}) where {T,L}
    ptrs = Ptr{Cvoid}[]
    for x in lower
        x isa TTVector{T,L} || throw(ArgumentError(
            "lower states must be TTVector{$T,$L}, got $(typeof(x))"))
        push!(ptrs, _ptr(x))
    end
    return ptrs
end

"""
    state_average_mals(H, x0, nroots; bond_dims = [0], noise = Float64[], weights = nothing,
                       homing = false, reorthonormalize = true, scale = 0.1, maxsubspace = 0,
                       restart_keep = 0, eps = 1e-6, conv_tol = 0.0, max_iter = 20, seed = 1234,
                       noise_type = :perturbative, verbose = false)
        -> (λs, states, history)

State-average two-site DMRG: the `nroots` lowest eigenpairs at once, in **one** train whose cores
are shared by every root except the orthogonality centre. Each bond is truncated once, against the
weighted density matrix `ρ = Σᵢ wᵢ BᵢBᵢᴴ`, so all roots live in a common basis — which is what
makes the returned states orthogonal by construction and the cost sublinear in `nroots`.

`weights` defaults to `1/nroots` each. Supplied weights must be finite, strictly positive,
**non-increasing** and sum to 1 (to 1e-12): those four clauses are load-bearing for the absolute
truncation tolerance, and each is a pointed refusal rather than a silent repair.

!!! warning "Ask for one more root than you need"
    The top root converges at the gap to the first **excluded** level, not to the one below it. On
    a fixture whose levels 2 and 3 are 0.025 apart, `nroots = 2` converges only linearly. Ask for
    `nroots + 1` and drop the last.

The shared basis inflates the rank grid — roughly `4·nroots` at the first interior bond — and the
returned states carry it: the driver never rounds them. Round each one yourself (see
[`round`](@ref Base.round(::QNTT.TTVector))) if you want its own rank.

`noise_type = :random` is **refused** here: the state-average truncation is defined only for
White density-matrix (perturbative) noise, whose enrichment is built per root and pooled with the
same weights. `nroots = 1` delegates to the single-state kernel and is bit-for-bit
[`mals`](@ref).

`history` is a vector of [`SaSweep`](@ref). Its `energy` is `Σᵢ wᵢ λᵢ` — the quantity the sweep
minimizes, not any single root — and `discarded_weight` is the pooled chop of the shared basis.

```julia
λs, xs, hist = QNTT.state_average_mals(H, x0, 3; eps = 1e-10, bond_dims = [0], max_iter = 30)
last(hist).discarded_weight
```

See also [`state_specific_mals`](@ref) and [`excited_states`](@ref).
"""
function state_average_mals(h::Hamiltonian{T,L}, x0::TTVector{T,L}, nroots::Integer;
                            bond_dims = [0], noise = Float64[], weights = nothing,
                            kwargs...) where {T,L}
    inner, outer = _split_mals_kwargs(kwargs)
    mals = setfields(defaults(MalsOptions); inner...)
    opts = setfields(defaults(StateAverageOptions); mals = mals, outer...)
    ws = weights === nothing ? Float64[] : Float64[w for w in weights]
    λs, ptrs, hist = _state_average_mals(L, T, _ptr(h), _ptr(x0), nroots, opts,
                                         ws, bond_dims, noise)
    return (λs, TTVector{T,L}[TTVector{T,L}(p) for p in ptrs], hist)
end

"""
    state_specific_mals(H, x0, lower; bond_dims = [0], noise = Float64[], mode = :penalty,
                        penalty = :auto, roots = 1, lower_energies = nothing, maxsubspace = 0,
                        restart_keep = 0, eps = 1e-6, conv_tol = 0.0, max_iter = 20, seed = 1234,
                        noise_type = :perturbative, verbose = false) -> (λ, x, history)

State-specific two-site DMRG: one root, driven away from the **fixed** states in `lower`. Each
local eigenproblem sees `H_eff + w Σⱼ pⱼ⟨pⱼ, ·⟩`, where `pⱼ` is the local image of `ψⱼ` built from
overlap environments — no `H·x` is ever materialized, and the cost per bond is one extra rank-`m`
correction.

`penalty = :auto` (the default) recomputes the weight each iteration as
`w = max(10(λ⊥ − minⱼEⱼ), 1)` on the energy of the component orthogonal to `lower`, and escalates
it tenfold if the state ever collapses back (`Σⱼoⱼ² > ½`). A number is used **verbatim and
unguarded**: a `w` below the gap to the target leaves the lower state as its own minimizer.
`mode = :project` searches the orthogonal complement instead and records `penalty == 0`; a bond
whose local space cannot host the projected directions is skipped rather than refused.

The lower states need not be normalized (`1/‖ψⱼ‖` is folded in) and are never mutated. If they are
themselves approximate, the target energy carries a **sign-indefinite** second-order error of
order `gap·|⟨ψⱼ,x⟩|²` that nothing in `history` can see — that is the reason
[`excited_states`](@ref) gates its lower states on an exact gradient. `lower_energies` supplies
the reference energies the auto rule needs, instead of computing them at entry.

`λ` is the **plain** energy `⟨x,Hx⟩/⟨x,x⟩`; `history[end].penalized` is the penalized Ritz value
the sweep actually minimized, and is never below it. With `lower` empty **and** `roots = 1` the
driver delegates to [`mals`](@ref) bit for bit, recording `penalty = 0` and `numops = 0`; at
`roots > 1` the block sweep runs even with no lower states, so neither holds. Unlike the state
average, this path accepts `noise_type = :random`.

```julia
λ0, g, _  = QNTT.mals(H, start; eps = 1e-10, bond_dims = [0])
λ1, x1, _ = QNTT.state_specific_mals(H, start, [g]; eps = 1e-10, bond_dims = [0])
abs(dot(x1, g))     # the residual overlap the penalty left behind
```
"""
function state_specific_mals(h::Hamiltonian{T,L}, x0::TTVector{T,L},
                             lower::AbstractVector{<:TTVector{T,L}};
                             bond_dims = [0], noise = Float64[], penalty = :auto,
                             lower_energies = nothing, kwargs...) where {T,L}
    rule, value = _penalty_fields(penalty)
    inner, outer = _split_mals_kwargs(kwargs)
    mals = setfields(defaults(MalsOptions); inner...)
    opts = setfields(defaults(StateSpecificOptions); mals = mals, penalty_rule = rule,
                     penalty_value = value, outer...)
    le = if lower_energies === nothing
        Float64[]
    else
        length(lower_energies) == length(lower) || throw(ArgumentError(
            "lower_energies must hold one entry per lower state ($(length(lower)))"))
        Float64[e for e in lower_energies]
    end
    λ, ptr, hist = _state_specific_mals(L, T, _ptr(h), _ptr(x0), _lower_ptrs(lower, T, L),
                                        opts, bond_dims, noise, le)
    return (λ, TTVector{T,L}(ptr), hist)
end

"""
    excited_states(H, x0; nroots = 2, lower = (), rank = 0, bond_dims = nothing,
                   noise = nothing, ss_noise = [0.0], weights = nothing, eps = 1e-10,
                   max_iter = 30, noise_type = :perturbative, seed = 1234, mode = :penalty,
                   penalty = :auto, ss_start = :sa, polish = true, keep_sa = false,
                   lower_grad_tol = 1e-6, trim_tol = 1e-6, trim_passes = 3, cg_eps = 1e-13,
                   cg_max_iter = 600, cg_restarts = 10, polish_tol = 1e-10, grad_gate = 1e-9,
                   overlap_tol = 1e-6, verbose = false) -> (λs, states, info)

The excited-state ladder: a state average for the starts, then, bottom-up and one root at a time,
a state-specific sweep against everything already fixed, a penalized Riemannian polish to a
critical point, and rank-trim passes — followed by Gram, swap and convergence gates. `lower` adds
already-known states the whole ladder is driven off.

`info` is a named tuple. The scalars are `licence`, `enriched`, `swaps` and `overlap_dev`; the
per-root vectors are `converged`, `stalled`, `swapped`, `gradnorms`, `penalties`, `passes`,
`sa_energies`, `ss_energies`, `j_ss`, `j_pol`; `overlaps` is the `(m+n)×(m+n)` absolute Gram over
`[lower; roots]`, `lower_gradnorms` the exact gate of each supplied lower state, and `sa_states`
the state-average ladder when `keep_sa = true`.

!!! warning "The licence does not certify completeness"
    `licence = all(converged) && overlap_dev ≤ overlap_tol && swaps == 0`. Every one of those is a
    statement about the roots you got — that they are critical, mutually orthogonal and in order.
    None of them says a level was not **skipped**: a shared truncation drops a sector carrying no
    weight, and H-driven noise cannot repopulate a sector `H` does not couple, so a licensed
    ladder can return the sixth level as its fifth root. Rerun with `nroots + 1` and compare the
    sets, or compare two starts.

At a **binding** rank cap the fixed-rank penalized minimizer keeps a residual overlap of order
`1/w`, so the `1e-6` Gram gate is simply unreachable there. Raise `rank` or the penalty — never
`overlap_tol`.

An empty schedule keeps the driver's own value, which is not uniformly "off": empty `weights` is
`1/nroots` and empty `bond_dims` the default ramp of `rank`, but `noise = nothing` is the built-in
nine-entry ramp `[3e-3 … 1e-6, 0.0]` and `ss_noise` defaults to `[0.0]`. Pass `noise = [0.0]` for
a noise-free ladder. A rank-1 determinant or CSF start is enriched by one noisy iteration rather
than refused (`info.enriched`), unlike [`state_average_mals`](@ref).

```julia
λs, xs, info = QNTT.excited_states(H, x0; nroots = 3, rank = 16)
info.licence, info.overlap_dev, info.gradnorms
```

On the SU(2) layer every root carries `2S` exactly, and `rank` / `bond_dims` are **reduced
multiplet** ranks.
"""
function excited_states(h::Hamiltonian{T,L}, x0::TTVector{T,L};
                        lower = (), weights = nothing, bond_dims = nothing, noise = nothing,
                        ss_noise = [0.0], penalty = :auto, kwargs...) where {T,L}
    rule, value = _penalty_fields(penalty)
    opts = setfields(defaults(ExcitedOptions); penalty_rule = rule, penalty_value = value,
                     kwargs...)
    ws = weights === nothing ? Float64[] : Float64[w for w in weights]
    bd = bond_dims === nothing ? Csize_t[] : Csize_t[b for b in bond_dims]
    ns = noise === nothing ? Float64[] : Float64[a for a in noise]
    lp = _lower_ptrs(lower, T, L)
    energies, roots, flat, rows, lg, gram, sa =
        _excited_states(L, T, _ptr(h), _ptr(x0), lp, opts, ws, bd, ns, ss_noise)
    m = length(lp) + Int(opts.nroots)
    info = (licence = flat.licence != 0,
            enriched = flat.enriched != 0,
            swaps = Int(flat.swaps),
            overlap_dev = flat.overlap_dev,
            overlaps = permutedims(reshape(gram, m, m)),
            converged = [r.converged != 0 for r in rows],
            stalled = [r.stalled != 0 for r in rows],
            swapped = [r.swapped != 0 for r in rows],
            gradnorms = [r.gradnorm for r in rows],
            penalties = [r.penalty for r in rows],
            passes = [Int(r.passes) for r in rows],
            sa_energies = [r.sa_energy for r in rows],
            ss_energies = [r.ss_energy for r in rows],
            j_ss = [r.j_ss for r in rows],
            j_pol = [r.j_pol for r in rows],
            lower_gradnorms = lg,
            sa_states = TTVector{T,L}[TTVector{T,L}(p) for p in sa])
    return (energies, TTVector{T,L}[TTVector{T,L}(p) for p in roots], info)
end

# =================================================================================================
# RPA / linear response
# =================================================================================================

"""
    rpa(H, x; k = 6, tol = 1e-8, max_iter = 800, seed = 1234) -> (ω, info)

Lowest `k` symplectic (RPA / linear-response) excitation energies `ω` at a STABLE point `x`: a
Julia-side convenience over [`rpa_signed`](@ref) with its defaults (the paired kernel, the `:ii`
side, no curvature regularization) that reads `ω = √ω²` off the signed spectrum. `max_iter` is the
budget in pair-equivalents (`(A, B)` apply pairs) and `tol` is relative in `ω` units
([`QNTT.SymplecticOptions`](@ref)). On the complex layers it asks [`rpa_signed`](@ref) for `2k`
realified values and keeps one `ω` per physical value. `info` reports `converged`, the basis size
(`iterations`) and the number of `(A, B)` apply pairs (`applies`; the per-mode `f_W` of
[`rpa_signed`](@ref) costs two more applies per mode, not counted).

A point with no real `ω` spectrum — an imaginary or complex mode, a de-excitation (Krein margin
`≤ 0`), a nonzero counting offset or a metric event — throws a [`QNTTError`](@ref) with code 5
(did not converge) and the signed spectrum in the message; [`rpa_signed`](@ref) returns it as
data. Convergence is reported, not enforced: gate on `info.converged`.

`x` should be a converged fixed-rank ground state (`riemannian` or `als` at the manifold rank).
The ω are fixed-rank linear-response estimates: they approximate, and only equal, exact
excitation energies when the manifold is the full space.
"""
function rpa(h::Hamiltonian{Float64,L}, x::TTVector{Float64,L};
             k::Integer = 6, seed::Integer = 1234, kwargs...) where {L}
    opts = setfields(defaults(SymplecticOptions); kwargs...)
    ω2, info = rpa_signed(h, x; k, seed, kernel = :paired, metric = :ii, curvature = :off,
                         max_iter = opts.max_iter, tol = opts.tol)
    return _rpa_omega(ω2, info, 1)
end

function rpa(h::Hamiltonian{ComplexF64,L}, x::TTVector{ComplexF64,L};
             k::Integer = 6, seed::Integer = 1234, kwargs...) where {L}
    opts = setfields(defaults(SymplecticOptions); kwargs...)
    ω2, info = rpa_signed(h, x; k = 2k, seed, kernel = :paired, max_iter = opts.max_iter,
                         tol = opts.tol)
    return _rpa_omega(ω2, info, 2)
end

# `rpa`'s gate: the predicate of the library's `RpaReadout::anomalous` read off `rpa_signed`'s info
# (an imaginary or complex mode, a real mode with a non-positive Krein margin, a nonzero counting
# offset or a metric event); `step = 2` keeps one of the two realified copies on the complex layers.
function _rpa_omega(ω2::Vector{Float64}, info, step::Int)
    r = info.report
    anomalous = any(<(0), ω2) || !isempty(info.complex) || any(<=(0), info.krein) ||
                something(r.neg_apb, 0) + something(r.neg_amb, 0) > 0 ||
                something(r.metric_events, 0) > 0
    if anomalous
        throw(QNTTError(5, "indefinite point, no real ω: signed ω² $(ω2), " *
                           "$(length(info.complex)) complex mode(s), Krein margins $(info.krein), " *
                           "report $(r); rpa_signed returns it as data"))
    end
    return (sqrt.(ω2[1:step:end]),
            (converged = info.converged, iterations = info.iterations, applies = info.applies))
end

"""
    rpa_dense(H, x) -> ω

The certified dense RPA spectrum over the whole tangent space at `x`: one Hessian apply per
tangent dimension, so this is a small-system cross-check of [`rpa`](@ref), not a production path.
Throws when `x` is not a stable minimizer (the diagnosis is in the message).
"""
rpa_dense(h::Hamiltonian{T,L}, x::TTVector{T,L}) where {T,L} =
    _rpa_dense(L, T, _ptr(h), _ptr(x))

"""
    sector_rpa(H_ref, H_out, x, phi; k = 4, readout = :tda, kernel = :paired, metric = :auto,
               curvature = :off, curvature_scale = 0.0, precond = :auto, precond_kappa = 0.0,
               tol = 1e-8, max_iter = 300, seed = 1234) -> (values, info)

Linear response INTO another point-group irrep from one typed reference: the lowest `k` values of
the flux-`phi` block of the Riemannian Hessian at `x`.

A typed fixed-rank state is irrep-closed — no ordinary tangent direction leaves its own irrep — so
[`rpa`](@ref) and [`rpa_dense`](@ref) can only ever report excitations WITHIN `Γ0`. Embedding `x`
in the point-group-untyped fixed-rank manifold splits its tangent space into one independent block
per irrep flux `phi`, and this is the `phi` block: response reaching `Γ0 ⊗ phi`, at typed cost.

# Arguments

- `H_ref` is the operator on `x`'s OWN space: it supplies `λ` and the curvature environments.
- `H_out` must be **built**, never re-stamped, on `x`'s space with `target = xor(Γ0, phi)`, from
  the SAME integrals, three-body term and `eps` as `H_ref` — the assembly automaton reads `target`
  for its caps and its bond-`d` collapse. Only the space stamp can be checked (a compiled
  Hamiltonian does not carry its inputs), so identical integrals are your obligation:

  ```julia
  s  = QNTT.QnSpace(d, nup, ndn; point_group = :C2v, orbsym, target = Γ0)
  H  = QNTT.Hamiltonian(s, t, v)
  Ho = QNTT.Hamiltonian(QNTT.QnSpace(d, nup, ndn; point_group = :C2v, orbsym,
                                     target = xor(Γ0, phi)), t, v)
  vals, info = QNTT.sector_rpa(H, Ho, x, phi)
  ```

- `phi != 0` is required: `phi = 0` is the in-sector problem, and [`rpa`](@ref) /
  [`rpa_dense`](@ref) answer it. Fluxes outside the state's group are rejected.
- `readout` is `:tda` — `eig(A^φ)`, an **energy** relative to `λ` — or `:rpa`, the **signed `ω²`**
  of the pair `(A^φ, B^φ)`. Different quantities in different units; see
  [`QNTT.SECTOR_READOUTS`](@ref). `kernel` ([`QNTT.RPA_KERNEL_CODES`](@ref)) picks the `:rpa`
  solve: `:paired` (the default, the paired-basis Davidson: no Hessian block is a metric) or
  `:signed` (the signed W-metric Davidson). `metric` ([`QNTT.RPA_METRIC_CODES`](@ref)) names the
  side `q = X ∓ Y` the Weingarten fraction reads and, under `kernel = :signed`, also the block that
  carries the W-metric — `:rr` the swap, `:ii` and `:auto` the `H^ii` lane, with no probe and no
  retry. Both are ignored on `:tda`.
- `curvature` / `curvature_scale` ([`QNTT.CURVATURE_REG_CODES`](@ref)) regularize the channel's
  curvature term exactly as [`rpa_signed`](@ref)'s do (`:hard` = `P·B·P`, `:tikhonov` = `D·B·D`,
  the scale an absolute Schmidt value of `QNTT.schmidt(x)`): every `:rpa` output is then the
  regularized pencil's; on `:tda` the pair is validated and has no effect (`A` is never
  regularized). `precond` / `precond_kappa` ([`QNTT.SECTOR_PRECOND_CODES`](@ref)) pick the
  preconditioner of either readout: `:auto` (the default) is `:off` until a measured table promotes
  `:d0`, the fixed inverse of the channel's shifted diagonal. An unknown `curvature` or `precond`
  Symbol, or `:hard`/`:tikhonov` with a scale that is not finite and `> 0`, is an `ArgumentError`
  before the call; an integer code is passed through and refused by the library
  ([`QNTTError`](@ref)). Every other keyword is a field of [`QNTT.SectorOptions`](@ref).

# Reading the answer

`info = (; readout, kernel, dim, metric, converged, iterations, applies, negatives_certified, f_w,
residuals, krein, complex, metric_indefinite, report)`.

- **`info.dim` is `dim T^φ`, NOT the size of the `Γ0 ⊗ phi` sector.** The shifted tangent space
  borrows `x`'s suffix bases at the relabeled sectors, so it is generally a proper subspace of that
  sector — for group reasons, even at full typed rank. The values are Ritz values compressed to
  `T^φ`, i.e. upper bounds by Cauchy interlacing, and are the sector's own levels only when `dim`
  equals its determinant count. Compare the two before quoting a number as an excitation energy.
- **An empty `values` is an answer, not an error.** It arises two ways, separated by
  `(converged, dim)`: a structurally unreachable channel (`dim == 0`, `converged = true`, no
  applies — there is nothing to converge) and a live channel whose random start block the
  projector annihilated (`dim > 0`, `converged = false`, no applies — a span statement, re-seed).
- On `:rpa` neither kernel refuses on the metric: the list is every imaginary mode plus the `k`
  lowest real ones (`length(values) ≥ k` is possible), and per entry `f_w` (the Weingarten
  fraction `|⟨q,Bq⟩| / |⟨q,Aq⟩|` on the side `metric` names), `residuals` (the pencil residual
  `‖Hz − ωΣz‖/‖z‖`) and `krein` (the Krein margin `(‖X‖² − ‖Y‖²)/(‖X‖² + ‖Y‖²)`, `0` for a
  negative `ω²`); `complex` holds the WANTED complex modes `(; re, im, residual)` — those whose
  `Re ω²` lies inside the reported window, not every complex pair of the projected pencil. A
  **negative `ω²` is a resonance** `ω = i√|ω²|` — publish one only under
  `info.negatives_certified`. A positive `ω²` with a negative margin is a DE-EXCITATION to a lower
  state of the target irrep, not an excitation. `info.report` is
  `(; neg_apb, neg_amb, metric_events, n_negative_basis, general_path)` as on
  [`rpa_signed`](@ref): the paired kernel's counting offset, the signed kernel's metric report,
  `nothing` where the kernel that ran has no such number. **Any anomaly — a nonzero offset, a
  metric event (`info.metric_indefinite`, signed only), a negative margin, an imaginary or complex
  mode — means `converged` does not license the values.** `info.kernel` is the request;
  `info.metric` the side used (the signed lane), or `:auto` where a signed run returned before
  choosing one. All per-mode fields are empty on `:tda`, and `report` is all `nothing`.
- A negative `:tda` value says `x` is not the lowest state of `Γ0 ⊗ phi`. That is information too:
  criticality is not stability.

Pass `k >= 1`: `k = 0` reports the channel's true `dim` with an empty answer and `converged = true`,
the one case where `converged` does not mean "the channel is fine".

Typed to `Float64` and the abelian layer.
"""
function sector_rpa(h_ref::Hamiltonian{Float64,Abelian}, h_out::Hamiltonian{Float64,Abelian},
                    x::TTVector{Float64,Abelian}, phi::Integer;
                    k::Integer = 4, readout::Symbol = :tda,
                    kernel::Union{Symbol,Integer} = :paired, metric::Symbol = :auto,
                    curvature::Union{Symbol,Integer} = :off, curvature_scale::Real = 0.0,
                    precond::Union{Symbol,Integer} = :auto, precond_kappa::Real = 0.0,
                    seed::Integer = 1234, kwargs...)
    kcode = _code(RPA_KERNEL_CODES, kernel, "kernel")
    ckind, cval = _curvature_fields(curvature, curvature_scale)
    pcode, pkappa = _precond_fields(precond, precond_kappa)
    opts = setfields(defaults(SectorOptions); readout, metric, kernel = kcode, kwargs...)
    vals, dim, met, conv, iters, applies, cert, f_w, residuals, krein, cplx, report =
        _sector_rpa(Abelian, _ptr(h_ref), _ptr(h_out), _ptr(x), phi, Csize_t(k), opts,
                    ckind, cval, pcode, pkappa, UInt64(seed))
    rep = _rpa_report(report)
    return (vals, (; readout, kernel = get(RPA_KERNELS, Cint(kcode), kcode), dim,
                     metric = RPA_METRICS[Cint(met)], converged = conv,
                     iterations = iters, applies = applies, negatives_certified = cert, f_w,
                     residuals, krein, complex = cplx,
                     metric_indefinite = something(rep.metric_events, 0) > 0, report = rep))
end

"""
    sector_rpa_dense(H_ref, H_out, x, phi) -> (ω², info)

The certified dense oracle of the flux-`phi` block: the signed `ω²` over the WHOLE shifted tangent
basis, one Hessian-class apply pair per tangent dimension. A small-system cross-check of
[`sector_rpa`](@ref)`(...; readout = :rpa, curvature = :off)` — the unregularized pencil; it takes
no curvature option — not a production path, and it solves twice, once to size the answer and
once to fill it.

`H_ref`, `H_out` and `phi` carry exactly the contract [`sector_rpa`](@ref) documents.

`info = (; dim, status, metric)`. `status` is `:stable` (both Hessian blocks positive definite,
every `ω² > 0`) or `:resonant` (exactly one block PD — the signed `ω²` is still real, and a
negative entry is a resonance `ω = i√|ω²|`); a caller testing `status !== :stable` must handle
`:resonant`. The third arm of the underlying tri-state, both blocks indefinite, is where no real
`ω²` exists at all: it throws [`QNTTError`](@ref) with the diagnosis, and it is the reason this
function's contract differs from [`rpa_dense`](@ref)'s — see [`QNTT.SECTOR_STATUS`](@ref).
`info.metric` names the positive-definite block on the resonant arm and is `:auto` otherwise, where
no metric was chosen.

Typed to `Float64` and the abelian layer.
"""
function sector_rpa_dense(h_ref::Hamiltonian{Float64,Abelian},
                          h_out::Hamiltonian{Float64,Abelian},
                          x::TTVector{Float64,Abelian}, phi::Integer)
    w2, dim, stable, met = _sector_rpa_dense(Abelian, _ptr(h_ref), _ptr(h_out), _ptr(x), phi)
    return (w2, (; dim, status = SECTOR_STATUS[Cint(stable)],
                   metric = RPA_METRICS[Cint(met)]))
end

# --- cross-sector observables --------------------------------------------------------------------

# One to three symmetric `norb × norb` matrices, normalized to a Vector. The COUNT is not checked
# here: `n_props > 3` is the library's own refusal (the oscillator-strength kernel is a three-row
# kernel) and it must cross as a `QNTTError` naming the component, not as an ArgumentError that
# says something else.
function _prop_list(props)
    ms = props isa AbstractMatrix ? [props] : collect(props)
    isempty(ms) && throw(ArgumentError("pass at least one property matrix"))
    n = size(ms[1], 1)
    for (i, p) in enumerate(ms)
        size(p) == (n, n) ||
            throw(ArgumentError("property $i is $(size(p)); every component must be the same \
                                 square norb × norb matrix"))
    end
    return ms
end

# The `Option<f64>` wire pair behind the single `sym_assert_rel` keyword. `nothing` (the default)
# is the flag CLEAR, which is what makes the library recompute `100·tol` from the `tol` actually
# passed instead of freezing it at the wire.
_sym_assert(::Nothing) = (has_sym_assert_rel = Cint(0),)
function _sym_assert(v::Real)
    (isfinite(v) && v > 0) ||
        throw(ArgumentError("sym_assert_rel must be finite and positive, got $v"))
    return (has_sym_assert_rel = Cint(1), sym_assert_rel = Float64(v))
end

"""
    sector_alpha(H_ref, H_out, x, phi, props; labels = nothing, tol = 1e-8, max_iter = 500,
                 tt_grad_tol = 1e-6, sym_assert_rel = nothing, route = :materialized)
        -> (alpha, info)

Static polarizability of ONE shifted irrep channel: `α^φ_μν = 2⟨p_μ, (H^{rr,φ})⁻¹ p_ν⟩` at the
typed reference `x`, for the one-body observables `props`.

`props` is 1 to 3 symmetric `norb × norb` matrices (a `Vector{<:AbstractMatrix}`, or one matrix).
Each is `μ̂ = Σ_{p,q} m[p,q]·E_pq`, spin-summed, and each must be the FULL symmetric matrix — both
triangles, not one representative per pair. **Supply your own hermitian matrices**: no shipped
fixture with a nontrivial point group carries a dipole triple, and the tests build synthetic
symmetry-typed operators from the fixture's own site-irrep table.

**The irrep flux is derived from the matrix**, not from an axis table — which irrep `z` carries
depends on the molecular orientation the integral generator chose, not on the group. A component
whose flux is not `phi` is a structural ZERO row and column with no work paid; a component whose
matrix carries more than one flux class raises [`QNTTError`](@ref) — split it by class and call
once per channel.

`H_ref`, `H_out` and `phi` carry exactly the contract [`sector_rpa`](@ref) documents, and `x` must
be normalized (`α`, `T_n` and `λ` all scale with the reference, so a non-unit `x` is refused).

Returns `alpha`, an `n × n` symmetric matrix in the order `props` was given, and
`info = (; alpha_defect, dim, phi, target, sigma_min, capture_defect, grad_norm, rel_residual,
iterations)`. `alpha` is a `1×1` or `2×2` principal sub-block of the Cartesian tensor embedded in
your shape: only the components whose flux is `phi` are visible in this channel, and summing the
channels — plus the in-sector block, which [`rpa`](@ref)'s problem answers and this one does not —
is what gives the full tensor. Read `info.capture_defect` and `info.sigma_min` with every number:
the first is how much of `μ̂x` the channel's borrowed-suffix tangent space MISSED (`O(1)` on a
proper-subspace channel, exactly `1` out of flux), the second the base-point conditioning.
`info.rel_residual` is the FRESH relative residual, one extra apply after each solve.

A refusal throws [`QNTTError`](@ref): the reference is not TT-critical, `H^rr` is indefinite on the
channel (then `x` is not the lowest state of `Γ0 ⊕ phi` — physics, not failure), or a solve hit its
cap. It is never a shifted or partial number. WHICH refusal is on the exception:
`err.refusal` is the wire code [`QNTT.SECTOR_REFUSALS`](@ref) names (`:not_tt_critical`,
`:hessian_indefinite`, `:solve_not_converged`), so branch on it instead of parsing `err.msg`. A
CALLER error — a non-symmetric matrix, more than three components, a non-unit `x`, a `route` other
than `:materialized` — throws with `err.refusal == 0` (`:none`): it is a statement about the
arguments, not about the state.

Every other keyword is a field of [`QNTT.SectorAlphaOptions`](@ref).

Typed to `Float64` and the abelian layer.
"""
function sector_alpha(h_ref::Hamiltonian{Float64,Abelian}, h_out::Hamiltonian{Float64,Abelian},
                      x::TTVector{Float64,Abelian}, phi::Integer, props;
                      labels = nothing, sym_assert_rel = nothing, kwargs...)
    ms = _prop_list(props)
    m = vcat((_rowmajor(Matrix{Float64}(p)) for p in ms)...)
    opts = setfields(defaults(SectorAlphaOptions); _sym_assert(sym_assert_rel)..., kwargs...)
    a, pr, it, info, _ = _sector_alpha(Abelian, _ptr(h_ref), _ptr(h_out), _ptr(x), phi, m,
                                       length(ms), size(ms[1], 1), labels, opts)
    n = length(ms)
    return (permutedims(reshape(a, n, n)),
            (; alpha_defect = info.alpha_defect, dim = Int(info.dim), phi = Int(info.phi),
               target = Int(info.target), sigma_min = info.sigma_min,
               capture_defect = pr[1:3:end], grad_norm = pr[2:3:end],
               rel_residual = pr[3:3:end], iterations = it))
end

"""
    sector_transition_moments(H_ref, H_out, x, phi, props; labels = nothing, dim = nothing,
                              tt_grad_tol = 1e-6, degeneracy_tol = 1e-8, psd_tol = 1e-10,
                              route = :materialized) -> (omega, moments, info)

Transition moments and oscillator strengths of ONE shifted irrep channel, DENSE route: the
`(A^φ, B^φ)` Grams over the whole shifted tangent basis, then a signed reduced-Williamson
reduction. One Hessian-class apply pair per tangent dimension plus `O(dim³)` dense work, so a
small-system certified path, not a production one — the same bargain
[`sector_rpa_dense`](@ref) makes.

`props`, `H_ref`, `H_out`, `phi` and the flux-derivation rule are exactly
[`sector_alpha`](@ref)'s; **supply your own hermitian matrices**.

`omega` holds the REAL frequencies, ascending, and `moments` is an `n_props × length(omega)`
matrix. **Only `|T_n|`, same-`n` products across two components, and cluster sums are
observable** — a bare signed `T_n` is gauge, and inside a degenerate cluster (`info.clusters`)
only `Σ_{n∈cluster} T_n T_n'` is defined at all.

`info = (; oscillator, alpha_sos, capture_defect, completeness_defect, sum_rule, sum_rule_si,
orth_error, sigma_min, clusters, n_modes, n_neg, unstable_omega2, dim, phi, target, metric,
complete_basis)`.

- `oscillator` is `f_n = (2/3)ω_n Σ_c T_n(c)²`; with fewer than three components it is a PARTIAL
  sum and the `2/3` is **not** rescaled.
- `alpha_sos` is the sum-over-states reconstruction of [`sector_alpha`](@ref)'s tensor — an
  algebraic identity at any rank, hence the mutual cross-check, with `completeness_defect` and
  `orth_error` as its error budget. It is `nothing` whenever `n_neg > 0`: the resolution of the
  identity is deficient by the imaginary modes' rank, and nothing partial is returned silently.
- `sum_rule` is the pair `(Σ_n T_n², ‖P^φ(μ̂x)‖²)` per component: EQUAL when `B^φ ≡ 0`, and
  otherwise a defect of **either** sign — not an inequality, and not Thomas–Reiche–Kuhn, which
  needs the true position operator in a complete basis and is a diagnostic here, never an oracle.
  `sum_rule_si` is its `S_i`-side twin.
- `clusters` is a `Vector{UnitRange{Int}}` partitioning `1:length(omega)`, singletons included —
  converted to **1-based Julia ranges** from the C half-open `[lo,hi)` pairs, the one place these
  bindings change an index convention.
- `unstable_omega2` holds the signed `ω² < 0` of the imaginary modes: `ω = i√|ω²|`, and it is
  **never** in `omega`.
- `metric` names the Hessian block that carried the PD metric, and is `:auto` on a structurally
  empty channel, where none was resolved.

`dim = nothing` (the default) — and only that — takes the two-pass route and therefore pays the
dense solve TWICE, once to size the answer and once to fill it. `dim = 0` is a KNOWN dimension (a
structurally empty channel), not an absent one, and takes the one-solve route. Passing the `dim`
from
[`sector_rpa`](@ref)`(...; k = 0)` — free, no applies, and an upper bound on `n_modes`, `n_neg`
and the cluster count alike — takes the one-solve route and returns the same numbers exactly.

A refusal throws [`QNTTError`](@ref): the reference is not TT-critical, or both Hessian blocks are
indefinite and the channel pencil admits no real symplectic basis at all — then there is no partial
answer either, so no `ω`, no `T` and no `f` are published. `err.refusal` names which
([`QNTT.SECTOR_REFUSALS`](@ref): `:not_tt_critical`, `:symplectic_unstable`), and is `0` (`:none`)
on a caller error.

Every other keyword is a field of [`QNTT.SectorMomentOptions`](@ref).

Typed to `Float64` and the abelian layer.
"""
function sector_transition_moments(h_ref::Hamiltonian{Float64,Abelian},
                                   h_out::Hamiltonian{Float64,Abelian},
                                   x::TTVector{Float64,Abelian}, phi::Integer, props;
                                   labels = nothing, dim = nothing, kwargs...)
    ms = _prop_list(props)
    m = vcat((_rowmajor(Matrix{Float64}(p)) for p in ms)...)
    opts = setfields(defaults(SectorMomentOptions); kwargs...)
    np = length(ms)
    om, mo, osc, sos, pr, un, cl, info, _ =
        _sector_transition_moments(Abelian, _ptr(h_ref), _ptr(h_out), _ptr(x), phi, m, np,
                                   size(ms[1], 1), labels, opts,
                                   dim === nothing ? nothing : Int(dim))
    nm = Int(info.n_modes)
    moments = permutedims(reshape(mo, nm, np))
    clusters = [(cl[2i-1]+1):cl[2i] for i in 1:Int(info.n_clusters)]
    return (om, moments,
            (; oscillator = osc,
               alpha_sos = info.has_alpha_sos != 0 ? permutedims(reshape(sos, np, np)) : nothing,
               capture_defect = pr[1:5:end], completeness_defect = pr[2:5:end],
               sum_rule = collect(zip(pr[3:5:end], pr[4:5:end])), sum_rule_si = pr[5:5:end],
               orth_error = info.orth_error, sigma_min = info.sigma_min, clusters = clusters,
               n_modes = nm, n_neg = Int(info.n_neg), unstable_omega2 = un,
               dim = Int(info.dim), phi = Int(info.phi), target = Int(info.target),
               metric = RPA_METRICS[Cint(info.metric)],
               complete_basis = info.complete_basis != 0))
end

# =================================================================================================
# Positive-definiteness probes and the complex seed
# =================================================================================================

"""
    pd_probe(H, x; part = :ii, budget = 40, seed = 1234)
        -> (verdict, lambda, residual, is_pd, applies)

Smallest-eigenvalue probe of the radially-deflated Riemannian Hessian at `x`.

`part` is `:rr` (`A + B`, the fixed-rank energy Hessian — indefinite means `x` is a saddle) or
`:ii` (`A − B` — indefinite means a physical instability against complex perturbations, and the
witness is the direction to complexify along). On a `ComplexF64` state there is one object, `𝔥`,
reached with `:rr`; `:ii` throws there. See [`QNTT.HESS_PARTS`](@ref).

Returns a `NamedTuple`, matching the `info` tuple [`rpa`](@ref) returns:

- `verdict` — `:witness_negative` (a PROOF: a fresh-apply Rayleigh quotient below zero, valid
  however unconverged the Davidson was), `:likely_pd`, or `:inconclusive`;
- `lambda` — that Rayleigh quotient;
- `residual` — the Davidson residual, `NaN` on `:witness_negative` where it means nothing. **Read
  it next to `lambda` every time**: a `:likely_pd` whose residual is within an order of `lambda`
  says nothing at all;
- `is_pd` — `true` / `false` / `nothing`, the tri-state. `nothing` is `:inconclusive` and must not
  be collapsed into either boolean;
- `applies` — Hessian applies actually spent.

**A negative verdict is not an error and does not throw** — it is the goal of the `:ii` probe.

`budget` caps Davidson iterations, i.e. roughly applies, and is not a formality: at a tangent
dimension of ~2·10⁴ a budget of 40 cannot converge, and a `:likely_pd` there can report its
margin an order of magnitude too high. Raise it until `residual` is small against `lambda`.
"""
function pd_probe(h::Hamiltonian{T,L}, x::TTVector{T,L};
                  part::Symbol = :ii, budget::Integer = 40,
                  seed::Integer = 1234) where {T,L}
    code, λ, res, pd, applies =
        _pd_probe(L, T, _ptr(h), _ptr(x), _code(HESS_PARTS, part, "Hessian part"),
                  Csize_t(budget), UInt64(seed))
    return (verdict = PD_VERDICTS[Cint(code)], lambda = λ, residual = res,
            is_pd = pd > 0 ? true : (pd == 0 ? false : nothing), applies = applies)
end

"""
    to_c64(x) -> TTVector{ComplexF64,L}

Real → complex embedding of a state: the same space, the same rank grid, every coefficient copied
into the real part. Linear in the coefficient count.

The orthogonality centre is carried over, because it stays true — a real column-orthonormal frame
is still orthonormal under the Hermitian inner product — so the cast state keeps the centre-core
norm shortcut and does not pay a canonicalization sweep to rediscover what it already knew.

Only this direction exists; `ComplexF64 → Float64` would have to discard an imaginary part.
"""
to_c64(x::TTVector{Float64,L}) where {L} = TTVector{ComplexF64,L}(_vec_to_c64(L, _ptr(x)))

"""
    complex_seed(H, x; eps = 1e-3, budget = 40, seed = 1234) -> (x_c, lambda)

The step that leaves the real manifold: probe `H^ii` at the REAL state `x`, take the proven
negative-curvature witness `w`, and return `normalize(retract(x, i·ε·ŵ))` as a
`TTVector{ComplexF64}` on `x`'s own rank grid, together with the witness's Rayleigh quotient
`lambda` (the `λ` in the seed's predicted energy change `λ·ε²` — a DECREASE, since `λ < 0`,
and with no ½: along a g-orthonormal direction the operator is already the half-second-derivative).

**Throws** [`QNTTError`](@ref) (`QNTT_NOT_CONVERGED`), rather than returning a state, when the
probe finds no negative direction — whether because none exists (`:likely_pd`) or because the
budget ran out (`:inconclusive`); the message says which. That refusal is the point: seeding along
an unproven direction is the null result, and the complex polish then quietly returns to the real
state.

Verify afterwards with `ρ = |(x,x)| / ⟨x,x⟩` — the bilinear, unconjugated form over the conjugated
one. `ρ = 1` means the run stayed real and every number downstream is the real answer in disguise.
"""
function complex_seed(h::Hamiltonian{Float64,L}, x::TTVector{Float64,L};
                      eps::Real = 1e-3, budget::Integer = 40,
                      seed::Integer = 1234) where {L}
    ptr, λ = _complex_seed(L, _ptr(h), _ptr(x), Float64(eps), Csize_t(budget), UInt64(seed))
    return (TTVector{ComplexF64,L}(ptr), λ)
end

# =================================================================================================
# The joint TT ⊕ orbital stack (CASSCF-style), the saddle seed and GAD
# =================================================================================================

"""
    cas_reference_state(H, cas; m_active = 0, seed = 1234) -> (λ0, x0)

The sector-pinned CAS reference state of `cas = (n_core, n_active, n_virt)` (site order: a
doubly-occupied core prefix, the active window, an empty virtual suffix) on its pinned rank grid,
with the active window at reduced rank `m_active` (`0` = full active rank). Returns the
ELECTRONIC Rayleigh quotient and the state — the starting point of [`casscf`](@ref).

!!! warning "Never MALS a pinned state"
    `ranks(x0) ≤ grid` is the postcondition (never equality); a two-site sweep regrows the pin.
    [`als`](@ref), and [`casscf`](@ref) with `preserve_grid = true`, are the sanctioned
    optimizers on it.
"""
function cas_reference_state(h::Hamiltonian{Float64,L}, cas; m_active::Integer = 0,
                             seed::Integer = 1234) where {L}
    λ, ptr = _cas_reference_state(L, _ptr(h), CasPartition(cas), Csize_t(m_active), UInt64(seed))
    return (λ, TTVector{Float64,L}(ptr))
end

"""
    casscf(data, x0, cas; orb_max_iter = 40, orb_grad = 1e-6, tt_grad_tol = 1e-6,
           orb_floor = 1e-8, preserve_grid = true, construction = :median, verbose = false)
        -> (λ, x, u, H_rot, info)

Alternating TT ⊕ orbital-rotation ground-state optimization (DMRG-SCF) from `x0` on the pinned
CAS manifold of `cas`, over the integrals `data` — whose orbitals are the INPUT basis.

**The results are meaningless without `u`.** `x`, `H_rot`, and everything computed from them
live in the rotated one-particle basis `u` (a `d × d` orthogonal matrix, `t' = uᵀ t u`), not the
input orbitals. `λ` is ELECTRONIC — add [`e0`](@ref)`(data)` for a total.

`info` is a `NamedTuple` mirroring the driver's diagnostics: `converged`, `rank_limited`,
`stalled_at_floor`, `iters`, `n_eff`, `refused_reason` (`nothing` or a `String`), the scalars
`sigma_min`, `u_orth_dev`, `g_dropped_final`, `ls_build_count`, `tt_legs`, `rejections`, and the
per-iteration series `e_history`, `g_orb_kept`, `g_orb_full`, `g_orb_dropped`, `tt_gradnorms`,
`orb_spectrum`. **The publication licence is `converged && !rank_limited`.**

The defaults are the flagship's (`examples/tutorial/step_13_casscf_response.rs`): the full orbital gradient is
gated at `orb_grad` too, and `construction = :median` (ignored on SU(2), which has one assembly).
`preserve_grid = true` (the default here, unlike the Rust driver) holds the pinned grid through
every TT leg; the pin-blind entry round would otherwise regrow it. The space, point group
included, is `x0`'s; `norb(data)` must equal its `d`.
"""
function casscf(data::Integrals{Float64}, x0::TTVector{Float64,L}, cas;
                preserve_grid = nothing, kwargs...) where {L}
    kw = Dict{Symbol,Any}(kwargs)
    preserve_grid === nothing || (kw[:tt_preserve_grid] = preserve_grid)
    opts = setfields(defaults(CasscfOptions); kw...)
    d = space(x0).d
    λ, xp, hp, uflat, ci, diag = _casscf(L, _ptr(data), _ptr(x0), CasPartition(cas), opts, d)
    x = TTVector{Float64,L}(xp)
    h = Hamiltonian{Float64,L}(hp)
    series = (; (k => _casscf_diag_series(diag, c) for (k, c) in pairs(_CASSCF_SERIES))...)
    reason = _casscf_diag_reason(diag)
    _casscf_diag_free(diag)
    u = permutedims(reshape(uflat, d, d), (2, 1))
    info = (; converged = ci.converged != 0, rank_limited = ci.rank_limited != 0,
              stalled_at_floor = ci.stalled_at_floor != 0, iters = Int(ci.iters),
              n_eff = Int(ci.n_eff), ls_build_count = Int(ci.ls_build_count),
              tt_legs = Int(ci.tt_legs), rejections = Int(ci.rejections),
              sigma_min = ci.sigma_min, u_orth_dev = ci.u_orth_dev,
              g_dropped_final = ci.g_dropped_final, refused_reason = reason,
              series...)
    return (λ, x, u, h, info)
end

"""
    joint_rpa(H_rot, x, cas; tt_grad_tol = 1e-6, orb_floor = 1e-8) -> (ω, resp)

The joint (TT ⊕ orbital) symplectic / RPA spectrum at the joint-critical point `(H_rot, x)`
returned by [`casscf`](@ref), over the pinned pairs of `cas`. `ω` is ascending on a `:stable`
point; on a `:resonant` one it carries `sign(ω²)·√|ω²|`, a negative entry being an imaginary pair
(a resonance, not a solver failure); it is empty on `:unstable` and `:refused`.

`resp` is `(; status, n_eff, n_modes, f_w, message)`: `status ∈ (:stable, :resonant, :unstable,
:refused)` — see [`JOINT_RPA_STATUS`](@ref) — `n_modes` the joint basis dimension, `n_eff` the
kept orbital coordinates, `f_w[j]` the Weingarten fraction `|⟨v,Bv⟩| / |⟨v,Av⟩|` of mode `j`
(`≪ 1` a physical excitation, `≈ 1` a manifold artifact; `NaN` on a resonant entry; empty on
`:unstable`/`:refused`), `message` a one-line diagnosis. **Nothing here throws**: `:refused`
means the point failed the joint context's own criticality / PD gate.
"""
function joint_rpa(h::Hamiltonian{Float64,L}, x::TTVector{Float64,L}, cas;
                   tt_grad_tol::Real = 1e-6, orb_floor::Real = 1e-8) where {L}
    ω, f_w, code, n_modes, n_eff, msg = _joint_rpa(L, _ptr(h), _ptr(x), CasPartition(cas),
                                                   Float64(tt_grad_tol), Float64(orb_floor))
    return (ω, (; status = JOINT_RPA_STATUS[Cint(code)], n_eff, n_modes, f_w, message = msg))
end

"""
    rpa_signed(H, x; k = 6, kernel = :paired, metric = :ii, curvature = :off,
               curvature_scale = 0.0, tol = 1e-8, max_iter = 800, seed = 1234) -> (ω², info)

Matrix-free RPA at a REAL reference **whatever its stability**. `kernel`
([`QNTT.RPA_KERNEL_CODES`](@ref)) is `:paired` (the default, the paired-basis Davidson: no Hessian
block is a metric) or `:signed` (the signed W-metric Davidson, on the lane `metric` names);
`max_iter` is the budget in pair-equivalents (`:paired` builds `max_iter` `(A, B)` apply pairs,
`:signed` `⌈max_iter/2⌉` basis vectors of two applies each) and `tol` is in `ω` units under both.
The readout is
the **signed** `ω²`, ascending: every imaginary mode the solve found plus the `k` lowest real ones,
so `k` counts excitations and `length(ω²) ≥ k` is possible. Positive entries are excitations
(`ω = √ω²`), negative ones resonances (`ω = i√|ω²|`), each certified by fresh applies. `metric`
([`QNTT.RPA_METRIC_CODES`](@ref)) names the side `q = X ∓ Y` the Weingarten fraction reads: `:ii`
(the default, and `:auto`) is `X − Y`, `:rr` is `X + Y`; under `kernel = :signed` it is also the
block that serves as the metric (`:auto` = `:ii`, no probe). An unknown `kernel` Symbol is an
`ArgumentError` before the call; an integer code passes through and the C side refuses a bad one
([`QNTTError`](@ref)).

`curvature` ([`QNTT.CURVATURE_REG_CODES`](@ref)) regularizes the curvature term `B` near the
border of the fixed-rank manifold, where it grows like the inverse of the smallest Schmidt value
and fills the bottom of the problem with curvature modes: `:off` (the default) is the exact
Riemannian Hessian; `:hard` solves the pencil with `B` replaced by `P·B·P`, `P` the column
projector off the right Schmidt directions with Schmidt value below `curvature_scale`; `:tikhonov`
uses `D·B·D`, `D = V diag(s²/(s² + ε²)) Vᵀ` with `ε = curvature_scale`. The scale is an absolute
Schmidt value — on the SU(2) layer the RAW `σ` of `QNTT.schmidt` (not `√weight·σ`). `A`
and the tangent space are untouched; every number returned (`ω²`, `f_w`, residuals, margins) is
the regularized pencil's. An unknown `curvature`, or `:hard`/`:tikhonov` with a scale that is not
finite and `> 0`, is an `ArgumentError` before the call (`:off` ignores the scale).

`info` fields: `metric ∈ (:ii, :rr)` (the side used), `negatives_certified` — **gate a published
`iκ` on it** — and per mode, each of `ω²`'s length: `f_w[j]`, the Weingarten fraction
`|⟨q,Bq⟩| / |⟨q,Aq⟩|` (two extra Hessian applies per mode: `≪ 1` a physical excitation, `≈ 1` a
manifold artifact of the fixed-rank constraint, on the `:ii` side `> 1` is algebraically
`ω² < 0`; a graded number beside `ω`, never a classifier), `residuals[j]` the pencil residual
`‖Hz − ωΣz‖/‖z‖` of the mode's amplitudes, `krein[j]` its Krein margin
`(‖X‖² − ‖Y‖²)/(‖X‖² + ‖Y‖²)` (`< 0` a de-excitation, `0` an imaginary mode). Solve level:
`complex`, the complex modes `(; re, im, residual)` of `ω²` (one member of each conjugate pair),
`kernel ∈ (:paired, :signed)`, the kernel that ran (from Julia, the request),
`report = (; neg_apb, neg_amb, metric_events, n_negative_basis, general_path)` — the paired
kernel's counting offset `n₋(A+B)`, `n₋(A−B)` of the projected pencil, the signed kernel's metric
report, `nothing` where the kernel that ran has no such number — and `converged`, `iterations`,
`applies`. Publish `ω` only from a converged solve with every `ω² > 0`, every margin positive, no
complex mode and a zero offset (or no metric event).
"""
function rpa_signed(h::Hamiltonian{Float64,L}, x::TTVector{Float64,L};
                    k::Integer = 6, kernel::Union{Symbol,Integer} = :paired,
                    metric::Union{Symbol,Integer} = :ii,
                    curvature::Union{Symbol,Integer} = :off, curvature_scale::Real = 0.0,
                    seed::Integer = 1234, kwargs...) where {L}
    kcode = _code(RPA_KERNEL_CODES, kernel, "kernel")
    ckind, cval = _curvature_fields(curvature, curvature_scale)
    opts = setfields(defaults(SymplecticOptions); kwargs...)
    ω2, f_w, residuals, krein, cplx, met, cert, conv, iters, applies, kernel, report =
        _rpa_signed(L, _ptr(h), _ptr(x), Csize_t(k), opts, kcode,
                    _code(RPA_METRIC_CODES, metric, "metric"), ckind, cval, UInt64(seed))
    return (ω2, (; metric = RPA_METRICS[Cint(met)], negatives_certified = cert, f_w, residuals,
                  krein, complex = cplx, kernel = RPA_KERNELS[Cint(kernel)],
                  report = _rpa_report(report), converged = conv, iterations = iters,
                  applies = applies))
end

"""
    rpa_signed(H::Hamiltonian{ComplexF64}, x; k = 6, kernel = :paired, tol = 1e-8,
               max_iter = 800, seed = 1234) -> (ω², info)

The complex-layer twin of the `Float64` method, on the REALIFIED tangent space: `kernel = :paired`
(the default) or `:signed`, which runs the W-metric Davidson on `:rr`, the layer's only metric. The
fields of `info` read as there, with three differences, each structural:

- no `metric` or `curvature` keyword, and `info.metric` is always `:rr` (`f_w` reads `X + Y`):
  `A + B` is the signed kernel's metric and `A − B` the other factor, orthogonally similar to it,
  so there is no block to choose (an indefinite point has both indefinite); the curvature
  regularization exists on real references only;
- **both realified copies** of every `ω²` come back, imaginary and complex ones included, so `k`
  counts realified values: ask `2k` for `k` physical ones ([`rpa`](@ref) collapses the pairs
  instead, its output being one `ω` per physical value);
- `f_w` is `|Re⟨q,Bq⟩| / |Re⟨q,Aq⟩|`, the realified quadratic forms.
"""
function rpa_signed(h::Hamiltonian{ComplexF64,L}, x::TTVector{ComplexF64,L};
                    k::Integer = 6, kernel::Union{Symbol,Integer} = :paired,
                    seed::Integer = 1234, kwargs...) where {L}
    kcode = _code(RPA_KERNEL_CODES, kernel, "kernel")
    opts = setfields(defaults(SymplecticOptions); kwargs...)
    ω2, f_w, residuals, krein, cplx, cert, conv, iters, applies, kernel, report =
        _rpa_signed_c64(L, _ptr(h), _ptr(x), Csize_t(k), opts, kcode, UInt64(seed))
    return (ω2, (; metric = :rr, negatives_certified = cert, f_w, residuals, krein,
                  complex = cplx, kernel = RPA_KERNELS[Cint(kernel)],
                  report = _rpa_report(report), converged = conv, iterations = iters,
                  applies = applies))
end

"""
    tda(H, x; k = 6, precond = :auto, seed = 1234, states = false, tol = 1e-8,
        max_iter = 300, ...) -> (θ, info)

The **same-sector Tamm–Dancoff** readout at a real reference `x`: the `k` lowest eigenpairs of the
Hessian's effective-Hamiltonian term `A = P_T(H − λ)P_T` alone (no curvature term `B`) on the
tangent space at `x` with the radial mode deflated, inside `x`'s own sector: its particle numbers,
its point-group irrep and, on the SU(2) layer, its spin (a singlet reference reaches singlet
excitations only). For another irrep use [`sector_rpa`](@ref) with `readout = :tda`.

How to read `θ` (ascending, at most `k` entries):

- `θ` is an **energy** in hartree **relative to `λ = ⟨x, Hx⟩`**, never `ω²`: the excited-state
  estimate is `λ + θ` (plus `QNTT.e0(data)` for a total energy; `λ` is
  [`rayleigh_quotient`](@ref)`(H, x)`, or the `λ` the ground-state solver returned). `λ + θⱼ` is,
  to the solve's accuracy, the energy of the explicit state `ι(ξⱼ)` that `states = true` returns.
- At a **critical** `x` (a converged fixed-rank ground state) the sorted list `{λ, λ + θⱼ}` bounds
  the lowest levels of `H` in the sector from above, entry by entry (MacDonald /
  Hylleraas–Undheim): the values are Ritz **upper bounds** from the tangent space, not exact
  levels, unless the manifold is the whole sector. Away from criticality the bound does not hold.
- A **negative `θ`** is a state below `x` inside its own sector and tangent span: `x` is **not**
  the lowest state of its sector — criticality is not stability. Each negative entry carries a
  fresh-apply certificate (`info.negatives_certified`).
- The readout reads no inverse Schmidt values, so no curvature regularization applies (the
  spectrum is independent of it bit for bit); at a stable minimizer the RPA `ω` of
  [`rpa_signed`](@ref) satisfy `ωⱼ ≤ θⱼ`.

`precond` ([`QNTT.TDA_PRECOND_CODES`](@ref)) is `:auto` (the default, run as `:diag` at `κ = 10`),
`:off`, `:diag` (capped by `precond_kappa`) or `:shifted`; an unknown Symbol is an `ArgumentError`
before the call, an integer code passes through and the C side refuses a bad one
([`QNTTError`](@ref)). `seed` fixes the `k + 2` random tangent warm starts, so a call is
reproducible. Every other keyword is a field of [`QNTT.TdaOptions`](@ref) — `tol` is an ABSOLUTE
residual tolerance in hartree.

`info` fields: `residuals[j]`, the fresh residual `‖Π A ξⱼ − θⱼ ξⱼ‖` of each returned pair;
`converged` (every requested root met `tol`); `negatives_certified`; `iterations`; `applies` (every
`A` apply, the fresh ones included); `space_dim`, the dimension solved on (`tangent_dim − 1`; at
most that many roots exist); `precond`, the arm that RAN ([`QNTT.TDA_PRECONDS`](@ref): `:auto`
resolved, `:off` where the `:diag` guard skipped a collapsed diagonal model); and `states`, a
`Vector` of the unit explicit Ritz states `ι(ξⱼ)` in `x`'s space (interior bonds doubled, so rank
`2r`; orthogonal to `x`) when `states = true`, empty otherwise. Fewer than `k` values come back
when `k` exceeds `space_dim` or the solve stagnates (`converged = false`).

```julia
λ, x = QNTT.mals(H, x0; bond_dims = [64], eps = 1e-8)
θ, info = QNTT.tda(H, x; k = 4, states = true)
E = λ .+ θ .+ QNTT.e0(data)           # excited-state estimates in x's sector
```
"""
function tda(h::Hamiltonian{Float64,L}, x::TTVector{Float64,L}; k::Integer = 6,
             precond::Union{Symbol,Integer} = :auto, seed::Integer = 1234, states::Bool = false,
             kwargs...) where {L}
    k ≥ 0 || throw(ArgumentError("k must be non-negative, got $k"))
    opts = setfields(defaults(TdaOptions); precond, kwargs...)
    θ, residuals, ptrs, conv, cert, iters, applies, dim, pre =
        _tda(L, _ptr(h), _ptr(x), Csize_t(k), opts, UInt64(seed), states)
    vs = TTVector{Float64,L}[TTVector{Float64,L}(p) for p in ptrs]
    return (θ, (; residuals, converged = conv, negatives_certified = cert, iterations = iters,
                 applies, space_dim = dim, precond = get(TDA_PRECONDS, Cint(pre), :unknown),
                 states = vs))
end

"""
    saddle_seed(H, x; eps = 1e-3, budget = 40, seed = 1234) -> (xk, λ_witness)

The GAD launch point: probe the real Hessian `A + B` at `x` and kick along its softest deflated
mode `ŵ`, `normalize(retract(x, eps·ŵ))`, on `x`'s own rank grid. `λ_witness` is that mode's
Rayleigh quotient. Unlike [`complex_seed`](@ref) a positive-definite verdict is not a refusal —
at a ground state the softest mode is exactly the kick wanted.
"""
function saddle_seed(h::Hamiltonian{Float64,L}, x::TTVector{Float64,L};
                     eps::Real = 1e-3, budget::Integer = 40, seed::Integer = 1234) where {L}
    ptr, λ = _saddle_seed(L, _ptr(h), _ptr(x), Float64(eps), Csize_t(budget), UInt64(seed))
    return (TTVector{Float64,L}(ptr), λ)
end

"""
    gad(H, x0; gradtol = 1e-8, max_steps = 600, k_davidson = 8, w_model = :switch,
        certify = true, probe_budget = 60, preserve_grid = false, seed = 1234,
        rank_adapt = :fixed, chi = 0, rank_eps = 1e-6, rank_period = 1,
        border_gain_max = 1.0, ...)
        -> (xe, info)

Gentlest-ascent dynamics from `x0` (prepare it with [`saddle_seed`](@ref)): an index-1 saddle of
the Rayleigh quotient on the rank manifold, i.e. an excited state. `w_model` is `:switch`
(the default), `:damped` or `:full`; every other keyword is a field of [`GadOptions`](@ref).
Pass `preserve_grid = true` on a sector-pinned (CAS) grid.

`rank_adapt` is `:fixed` (the default: the declared grid, ranks never above it), `:both` (the
accepted step is re-retracted through an energy-budget round of `x + α·F`, so ranks grow where
the step needs them — up to `chi`, a POOLED total bond-dimension cap (SU(2): reduced multiplets),
`0` = uncapped — and drop where a Schmidt value falls under `rank_eps`) or `:shrink` (the
accepted point is rounded to `rank_eps`, so the grid can only shrink — the manifold-border arm).
Adaptation happens between accepted steps, every `rank_period` of them. A sector at rank zero in
`x0`'s grid is never opened, so a CAS grid stays pinned under every mode.

`border_gain_max` is the manifold-border rule (Schmidt values → 0), where the Weingarten term
carries `Σ⁻¹` and the undamped certificate reads modes the state does not have: no certificate
runs when the endpoint has `min_schmidt < schmidt_floor` (with the Weingarten term live) or a
Weingarten gain `‖B·V‖/‖A·V‖` above `border_gain_max` (default `1.0`); the level is reported with
`border_verdict = true`, `index_estimate = nothing`, `converged = false`.

`info` is a `NamedTuple`: `converged` (criticality AND, with `certify`, the index-1 certificate),
`index1_certified`, `index_estimate` (`nothing` when no certificate ran or a probe was
inconclusive), `steps`, `applies`, `gnorm`, `lambda` (ELECTRONIC), `eig1` (`nothing`-able),
`v_overlap`, `b_gain_max`, `min_schmidt_min`, `schmidt_floor_hit`, `rejected_steps`,
`relatches`, `regime_switches`, `radial_overlap_max`, `branch_gap`, `witness_overlap`
(`nothing`-able: `|⟨w, V⟩|` between the certificate's own lowest mode and the tracked one),
`tracked_mode_is_lowest` (`false` with `index1_certified` means the saddle was reached on
another negative-curvature branch — the certificate classifies the point), `grid_changes`,
`border_verdict`, `b_gain_end` (`nothing`-able).
"""
function gad(h::Hamiltonian{Float64,L}, x0::TTVector{Float64,L}; kwargs...) where {L}
    opts = setfields(defaults(GadOptions); kwargs...)
    ptr, gi = _gad(L, _ptr(h), _ptr(x0), opts)
    info = (; converged = gi.converged != 0, index1_certified = gi.index1_certified != 0,
              index_estimate = gi.has_index_estimate != 0 ? Int(gi.index_estimate) : nothing,
              steps = Int(gi.steps), applies = Int(gi.applies), gnorm = gi.gnorm,
              lambda = gi.lambda, eig1 = gi.has_eig1 != 0 ? gi.eig1 : nothing,
              v_overlap = gi.v_overlap, b_gain_max = gi.b_gain_max,
              min_schmidt_min = gi.min_schmidt_min, schmidt_floor_hit = gi.schmidt_floor_hit != 0,
              rejected_steps = Int(gi.rejected_steps), relatches = Int(gi.relatches),
              regime_switches = Int(gi.regime_switches), radial_overlap_max = gi.radial_overlap_max,
              branch_gap = gi.branch_gap,
              witness_overlap = gi.has_witness_overlap != 0 ? gi.witness_overlap : nothing,
              tracked_mode_is_lowest = gi.tracked_mode_is_lowest != 0,
              grid_changes = Int(gi.grid_changes),
              border_verdict = gi.border_verdict != 0,
              b_gain_end = isnan(gi.b_gain_end) ? nothing : gi.b_gain_end)
    return (TTVector{Float64,L}(ptr), info)
end
