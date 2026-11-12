# Copyright (C) 2023-2026 Paul Cazeaux
#
# This file is part of qntt. qntt is free software: you can
# redistribute it and/or modify it under the terms of the GNU
# General Public License as published by the Free Software
# Foundation, either version 3 of the License, or (at your
# option) any later version. See LICENSE for details.

# =================================================================================================
# Handles
# =================================================================================================

"""
    TTVector{T,L}

A quantum-number-preserving tensor train. Freed automatically; `ptr` is an owned handle into the
Rust library.
"""
mutable struct TTVector{T,L<:Layer}
    ptr::Ptr{Cvoid}
    function TTVector{T,L}(ptr::Ptr{Cvoid}) where {T,L}
        x = new{T,L}(ptr)
        finalizer(v -> (v.ptr == C_NULL || _vec_free(L, T, v.ptr); v.ptr = C_NULL), x)
        return x
    end
end

"""
    Hamiltonian{T,L}

A compiled second-quantized operator (the operator-action automaton).
"""
mutable struct Hamiltonian{T,L<:Layer}
    ptr::Ptr{Cvoid}
    function Hamiltonian{T,L}(ptr::Ptr{Cvoid}) where {T,L}
        h = new{T,L}(ptr)
        finalizer(v -> (v.ptr == C_NULL || _ham_free(L, T, v.ptr); v.ptr = C_NULL), h)
        return h
    end
end

"""
    Integrals{T}

Parsed FCIDUMP contents: orbital/electron counts, `2·Sz`, the core energy and the one-/two-body
coefficient buffers.
"""
mutable struct Integrals{T}
    ptr::Ptr{Cvoid}
    function Integrals{T}(ptr::Ptr{Cvoid}) where {T}
        d = new{T}(ptr)
        finalizer(v -> (v.ptr == C_NULL || _fcidump_free(T, v.ptr); v.ptr = C_NULL), d)
        return d
    end
end

"""
    ThreeBody{T}

Sparse three-body (transcorrelated) coefficient tensor.
"""
mutable struct ThreeBody{T}
    ptr::Ptr{Cvoid}
    function ThreeBody{T}(ptr::Ptr{Cvoid}) where {T}
        w = new{T}(ptr)
        finalizer(v -> (v.ptr == C_NULL || _w_free(T, v.ptr); v.ptr = C_NULL), w)
        return w
    end
end

const _Handle = Union{TTVector,Hamiltonian,Integrals,ThreeBody}

function _ptr(h::_Handle)
    h.ptr == C_NULL && throw(QNTTError(2, "use of a freed handle"))
    return h.ptr
end

Base.eltype(::TTVector{T}) where {T} = T
Base.eltype(::Hamiltonian{T}) where {T} = T
layer(::TTVector{T,L}) where {T,L} = L
layer(::Hamiltonian{T,L}) where {T,L} = L

# =================================================================================================
# Building trains
# =================================================================================================

"""
    tt_zeros([T,] space) -> TTVector

All-zero train (every bond rank zero).
"""
tt_zeros(::Type{T}, s::Space{L}) where {T,L} =
    TTVector{T,L}(istrivial(s) ? _vec_zeros(L, T, s.d, s.a, s.b) :
                  _vec_zeros_pg(L, T, s.d, s.a, s.b, POINT_GROUPS[s.kind], s.orbsym, s.target))
tt_zeros(s::Space) = tt_zeros(Float64, s)

"""
    tt_from_ranks([T,] space, ranks) -> TTVector

Zero-filled train at prescribed per-bond rank grids (`length(ranks) == d+1`). Fill it with
[`setblock!`](@ref).
"""
function tt_from_ranks(::Type{T}, s::Space{L}, ranks) where {T,L}
    flat = _flatten_ranks(s, ranks)
    ptr = istrivial(s) ? _vec_from_ranks(L, T, s.d, s.a, s.b, flat) :
          _vec_from_ranks_pg(L, T, s.d, s.a, s.b, POINT_GROUPS[s.kind], s.orbsym, s.target,
                             flat)
    return TTVector{T,L}(ptr)
end
tt_from_ranks(s::Space, ranks) = tt_from_ranks(Float64, s, ranks)

"""
    tt_randn([T,] space, ranks; seed = 1234) -> TTVector

Gaussian random train at the prescribed ranks. The stream is fully determined by `seed`, so runs
are reproducible. With a point group, ranks at sectors the symmetry makes unreachable are
silently masked to zero, so a uniform grid still yields an irrep-pure train.
"""
function tt_randn(::Type{T}, s::Space{L}, ranks; seed::Integer = 1234) where {T,L}
    flat = _flatten_ranks(s, ranks)
    ptr = istrivial(s) ? _vec_randn(L, T, s.d, s.a, s.b, flat, UInt64(seed)) :
          _vec_randn_pg(L, T, s.d, s.a, s.b, POINT_GROUPS[s.kind], s.orbsym, s.target,
                        flat, UInt64(seed))
    return TTVector{T,L}(ptr)
end
tt_randn(s::Space, ranks; kwargs...) = tt_randn(Float64, s, ranks; kwargs...)

"""
    tt_state([T,] occ_up, occ_dn) -> TTVector
    tt_state([T,] occ_up, occ_dn, space) -> TTVector

Rank-1 abelian product state (a determinant) from per-site occupations. The two-argument form
derives the space from the occupation counts and carries no point group; the three-argument form
places the determinant in `space`, whose target irrep it must match.
"""
tt_state(::Type{T}, occ_up, occ_dn) where {T} =
    TTVector{T,Abelian}(_tt_state(T, occ_up, occ_dn))
tt_state(occ_up, occ_dn) = tt_state(Float64, occ_up, occ_dn)

function tt_state(::Type{T}, occ_up, occ_dn, s::Space{Abelian}) where {T}
    istrivial(s) && return tt_state(T, occ_up, occ_dn)
    return TTVector{T,Abelian}(
        _tt_state_pg(T, occ_up, occ_dn, POINT_GROUPS[s.kind], s.orbsym, s.target))
end
tt_state(occ_up, occ_dn, s::Space{Abelian}) = tt_state(Float64, occ_up, occ_dn, s)

"""
    tt_ones([T,] space) -> TTVector

The 0/1 indicator of the space's `(Nup, Ndn)` block — of its `(Nup, Ndn, target)` block when the
space carries a point group. Abelian layer only.
"""
tt_ones(::Type{T}, s::Space{Abelian}) where {T} =
    TTVector{T,Abelian}(_tt_ones_pg(T, s.d, s.a, s.b, POINT_GROUPS[s.kind], s.orbsym, s.target))
tt_ones(s::Space{Abelian}) = tt_ones(Float64, s)

"""
    tt_csf([T,] codes) -> TTVector
    tt_csf([T,] codes, space) -> TTVector

Rank-1 SU(2) configuration state function from per-site genealogical codes: `:empty`, `:up`
(the `S+½` branch), `:dn` (`S−½`), `:occ` (on-site singlet pair).

`:up`/`:dn` are **coupling branches, not electron spins** — both add exactly one electron.

The two-argument form derives the space from the walk and carries no point group; the
three-argument form places the CSF in `space`, whose target irrep it must match — the total irrep
is DERIVED from the walk and a `space` that disagrees is an error, so a caller cannot silently
land in a different symmetry block.
"""
function tt_csf(::Type{T}, codes) where {T}
    c = [_code(FAMILIES, v, "csf code") for v in codes]
    return TTVector{T,SU2}(_tt_csf(T, c))
end
tt_csf(codes) = tt_csf(Float64, codes)

function tt_csf(::Type{T}, codes, s::Space{SU2}) where {T}
    istrivial(s) && return tt_csf(T, codes)
    length(codes) == s.d ||
        throw(ArgumentError("need d = $(s.d) genealogical codes, got $(length(codes))"))
    c = [_code(FAMILIES, v, "csf code") for v in codes]
    return TTVector{T,SU2}(_tt_csf_pg(T, c, POINT_GROUPS[s.kind], s.orbsym, s.target))
end
tt_csf(codes, s::Space{SU2}) = tt_csf(Float64, codes, s)

# =================================================================================================
# Inspecting trains
# =================================================================================================

"""
    space(x) -> Space

The train's (or operator's) symmetry descriptor.

The point-group KIND is not recoverable from a built object — the library stores the packed
per-site table, not the kind, and `nbits` does not determine it (D2, C2v and C2h all have
`nbits = 2`). The descriptor therefore comes back labelled with the *conventional* kind of that
order, which is numerically identical and differs only in irrep names.
"""
space(x::TTVector{T,L}) where {T,L} =
    _relabel_space(L, _vec_space(L, T, _ptr(x)), _vec_point_group(L, T, _ptr(x)))
space(h::Hamiltonian{T,L}) where {T,L} =
    _relabel_space(L, _ham_space(L, T, _ptr(h)), _ham_point_group(L, T, _ptr(h)))

function _relabel_space(::Type{L}, (d, a, b), (nbits, target, irreps)) where {L}
    nbits == 0 && return Space{L}(d, a, b)
    kind = _pg_default_kind(1 << nbits)
    orbsym = UInt8[_pg_molpro_from_irrep(kind, ir) for ir in irreps]
    return Space{L}(d, a, b; point_group = POINT_GROUP_NAMES[kind],
                    orbsym = orbsym, target = target)
end

"""
    point_group(x) -> (nbits, target, irreps)

What the train's (or operator's) space actually stores: the number of irrep bits, the 0-based XOR
target, and the per-site 0-based XOR irreps. See [`space`](@ref) on why the kind is not among
them.
"""
point_group(x::TTVector{T,L}) where {T,L} = _vec_point_group(L, T, _ptr(x))
point_group(h::Hamiltonian{T,L}) where {T,L} = _ham_point_group(L, T, _ptr(h))

"""
    ranks(x) -> Vector{Matrix{Int}}
             -> Vector{Array{Int,3}}   (with a point group)

Per-bond rank grids, `g[a+1, b+1]` for sector `(a, b)` — `g[a+1, b+1, ir+1]` with a point group.
"""
function ranks(x::TTVector{T,L}) where {T,L}
    s = space(x)
    flat = istrivial(s) ? _vec_ranks_flat(L, T, _ptr(x)) : _vec_ranks_flat_pg(L, T, _ptr(x))
    return _unflatten_ranks(s, flat)
end

"""
    orthogonality_center(x) -> Union{Int,Nothing}

Site index of the orthogonality centre, or `nothing` when the train is not in canonical form.
"""
function orthogonality_center(x::TTVector{T,L}) where {T,L}
    k = _vec_ortho(L, T, _ptr(x))
    return k < 0 ? nothing : k
end

"""
    isvalid(x) -> Bool

Structural self-check (rank and sector consistency across cores).
"""
Base.isvalid(x::TTVector{T,L}) where {T,L} = _vec_check(L, T, _ptr(x))

"""
    nsites(x)              -> Int
    max_sector_rank(x)     -> Int
    max_bond_rank(x)       -> Int

Number of sites, largest single-sector rank, and largest summed-over-sectors bond dimension.
"""
nsites(x::TTVector{T,L}) where {T,L} = _vec_sizes(L, T, _ptr(x))[1]
max_sector_rank(x::TTVector{T,L}) where {T,L} = _vec_sizes(L, T, _ptr(x))[2]
max_bond_rank(x::TTVector{T,L}) where {T,L} = _vec_sizes(L, T, _ptr(x))[3]

function Base.show(io::IO, x::TTVector{T,L}) where {T,L}
    s = space(x)
    print(io, "TTVector{", T, ",", nameof(L), "}(", s,
          ", max bond dim ", max_bond_rank(x), ")")
end

"""
    getblock(x, site, family, sector) -> Matrix

One core block, column-major. `family` is `:empty`, `:up`, `:dn` or `:occ`; `sector` is the
`(a, b)` label of the block's ROW bond (`site`), or `(a, b, ir)` with a point group (a 2-tuple
then means irrep 0). An absent block comes back `0×0` — the common case once `|G| > 1`.
"""
function getblock(x::TTVector{T,L}, site::Integer, family, sector::Tuple) where {T,L}
    f = _code(FAMILIES, family, "family")
    s = space(x)
    istrivial(s) && return _block_get(L, T, _ptr(x), site, f, sector[1], sector[2])
    ir = length(sector) >= 3 ? sector[3] : 0
    return _block_get_pg(L, T, _ptr(x), site, f, sector[1], sector[2], ir)
end

"""
    setblock!(x, site, family, sector, M)

Overwrite one core block. The train must not be shared with another live handle (it is not,
unless you kept a value returned by an in-place op), and `M` must match the block's dimensions.
"""
function setblock!(x::TTVector{T,L}, site::Integer, family,
                   sector::Tuple, m::AbstractMatrix) where {T,L}
    f = _code(FAMILIES, family, "family")
    s = space(x)
    if istrivial(s)
        _block_set!(L, T, _ptr(x), site, f, sector[1], sector[2], m)
    else
        ir = length(sector) >= 3 ? sector[3] : 0
        _block_set_pg!(L, T, _ptr(x), site, f, sector[1], sector[2], ir, m)
    end
    return x
end

"""
    blocks(x, site) -> Dict{Tuple{Symbol,Tuple},Matrix}

Every present block of one core, keyed by `(family, sector)`. The sector is a 2-tuple without a
point group and a 3-tuple `(a, b, ir)` with one, matching [`bond_sectors`](@ref).
"""
function blocks(x::TTVector{T,L}, site::Integer) where {T,L}
    s = space(x)                     # resolved once: `getblock` would re-query it per block
    trivial = istrivial(s)
    out = Dict{Tuple{Symbol,Tuple},Matrix{T}}()
    for (fam, code) in FAMILIES, sec in bond_sectors(s, site)
        m = trivial ? _block_get(L, T, _ptr(x), site, code, sec[1], sec[2]) :
            _block_get_pg(L, T, _ptr(x), site, code, sec[1], sec[2], sec[3])
        isempty(m) || (out[(fam, sec)] = m)
    end
    return out
end

"""
    Array(x) -> Vector

Dense materialization in the `4^d` Jordan–Wigner basis (abelian layer only). Refuses to allocate
when `4^d` does not fit in memory — this is a small-system oracle, not a production path.
"""
Base.Array(x::TTVector{T,Abelian}) where {T} = _to_dense(T, _ptr(x))

# =================================================================================================
# Algebra
# =================================================================================================

LinearAlgebra.norm(x::TTVector{T,L}) where {T,L} = _vec_norm(L, T, _ptr(x))

LinearAlgebra.dot(x::TTVector{T,L}, y::TTVector{T,L}) where {T,L} =
    _vec_dot(L, T, _ptr(x), _ptr(y))

Base.:*(a::Number, x::TTVector{T,L}) where {T,L} =
    TTVector{T,L}(_vec_scaled(L, T, _ptr(x), convert(T, a)))
Base.:*(x::TTVector, a::Number) = a * x
Base.:/(x::TTVector{T}, a::Number) where {T} = inv(convert(T, a)) * x
Base.:-(x::TTVector{T}) where {T} = -one(T) * x

Base.:+(x::TTVector{T,L}, y::TTVector{T,L}) where {T,L} =
    TTVector{T,L}(_vec_add(L, T, _ptr(x), _ptr(y), Cint(1)))
Base.:-(x::TTVector{T,L}, y::TTVector{T,L}) where {T,L} =
    TTVector{T,L}(_vec_add(L, T, _ptr(x), _ptr(y), Cint(-1)))

"""
    round(x; eps = 1e-10, chi = 0, mode = :abs) -> TTVector

Deterministic SVD rounding (a method of `Base.round`). `chi` is a pooled total bond-dimension cap
(SU(2): reduced multiplets); `0` is uncapped. `mode` is `:abs` (absolute error budget) or `:rel`
(scaled by `‖x‖`).
"""
Base.round(x::TTVector{T,L}; eps::Real = 1e-10, chi::Integer = 0,
           mode::Symbol = :abs) where {T,L} =
    TTVector{T,L}(_vec_round(L, T, _ptr(x), Float64(eps), chi,
                             _code(ROUND_MODES, mode, "round mode")))

"""
    round!(x; eps = 1e-10, chi = 0, mode = :abs) -> TTVector

In-place rounding — the mutating twin of the `Base.round` method above. `chi` is a pooled total
bond-dimension cap (SU(2): reduced multiplets); `0` is uncapped. Requires an unshared train.
"""
function round!(x::TTVector{T,L}; eps::Real = 1e-10, chi::Integer = 0,
                mode::Symbol = :abs) where {T,L}
    _vec_round!(L, T, _ptr(x), Float64(eps), chi, _code(ROUND_MODES, mode, "round mode"))
    return x
end

"""
    left_orthonormalize!(x; keep_rank = false) -> Float64

Left-orthonormalization sweep; returns the product of the per-bond norm factors (i.e. `‖x‖`
before normalization).
"""
left_orthonormalize!(x::TTVector{T,L}; keep_rank::Bool = false) where {T,L} =
    _vec_left_orthonormalize!(L, T, _ptr(x), Cint(keep_rank))

_ptrs(xs) = Ptr{Cvoid}[_ptr(x) for x in xs]

"""
    round_sum(alpha, xs; tol = 1e-10) -> TTVector

`round(Σ αᵢ xᵢ, tol)` — the deterministic weighted sum, rounded once at the end rather than
after each addition.
"""
function round_sum(alpha::AbstractVector, xs::AbstractVector{TTVector{T,L}};
                   tol::Real = 1e-10) where {T,L}
    length(alpha) == length(xs) || throw(ArgumentError("alpha and xs must agree in length"))
    a = T[convert(T, v) for v in alpha]
    return TTVector{T,L}(_vec_round_sum(L, T, a, _ptrs(xs), Float64(tol)))
end

# =================================================================================================
# State diagnostics
# =================================================================================================

"""
    rho(x) -> Float64

`ρ = |(x,x)| / ⟨x,x⟩` — the **vacuity guard** of the complex-manifold pipeline. The numerator is
the bilinear, *unconjugated* self-contraction; the denominator the Hermitian one.

`ρ = 1` exactly when every coefficient shares one global phase, i.e. the state is real up to a
phase. A "complex" run that reports `ρ = 1` never left the real invariant subspace — complex
conjugation is a symmetry of the problem, so the real states are invariant under MALS/CG with real
environments — and every number downstream of it is the real answer in disguise. Print it after
every stage, not only at the end.

`norm(imag.(...))` is **not** a substitute: a global phase is unphysical, and rotating a real state
by `i` would send that quantity from 0 to `‖x‖` while `ρ` correctly stays 1. Identically 1 on a
`Float64` state, where the question does not arise.
"""
function rho(x::TTVector{T,L}) where {T,L}
    herm = real(_vec_dot(L, T, _ptr(x), _ptr(x)))
    herm > 0 || throw(ArgumentError("rho is undefined at the zero train"))
    return abs(_vec_dot_bilinear(L, T, _ptr(x), _ptr(x))) / herm
end

"""
    bilinear_dot(x, y) -> T

The unconjugated `(x, y)`, as against `dot(x, y) = ⟨x, y⟩`. See [`rho`](@ref) for why this is the
quantity that matters on the complex manifold. Over `Float64` it is `dot`.
"""
bilinear_dot(x::TTVector{T,L}, y::TTVector{T,L}) where {T,L} =
    _vec_dot_bilinear(L, T, _ptr(x), _ptr(y))

"""
    schmidt(x) -> Vector{Vector{Float64}}

Schmidt data at every **interior** bond `1:d-1`, as a `NamedTuple` of two parallel
`Vector{Vector{Float64}}`:

- `sigma` — the bond-gauge singular values, pooled over the bond's sectors and sorted descending.
  These are **reduced** values, never pre-multiplied by their weight: `σ` is the conditioning
  diagnostic that the Weingarten term's `Σ⁻¹` acts on and the trim criterion reads, and
  [`sigma_min`](@ref) reduces it.
- `weight` — each value's multiplet degeneracy.

One call; the whole table costs one canonicalization.

# The sum rule

At every interior bond of a unit-norm state,

    sum(w .* σ.^2) == 1

exactly (roundoff over the sum is the only slack). On the **abelian** layer every `w` is `1`, so
this is the ordinary `sum(σ.^2) == 1`. On the **SU(2)** layer
`w = centerweight(2s_col, 2S_total) = (2s_col + 1)/(2S_total + 1)`: each reduced multiplet
contributes its genuine degeneracy. It is the same weight the crate's `su2::chop` counts ranks on
(`√centerweight · σ`), so the truncation budget and this identity measure the same fixed-Sz energy.

The **unweighted** `sum(σ.^2)` is not a normalization check on SU(2), and its direction is not even
fixed. At `2S = 0` every weight is `≥ 1`, so it sits below 1 (e.g. 0.52–0.98 on a random
singlet, 0.9895–0.9994 on the water 6-31G ground state); at `2S > 0` sectors with
`2s_col < 2S_total` weigh less than 1 and it runs above (e.g. 1.47–2.99 at `2S = 2`,
1.25–4.81 at `2S = 4`).
"""
schmidt(x::TTVector{T,L}) where {T,L} = _vec_schmidt(L, T, _ptr(x))

"""
    sigma_min(x) -> Float64

Smallest Schmidt value over all interior bonds — the **manifold-boundary diagnostic**.

The Weingarten curvature scales like `‖r⊥‖/σ_min`, so a collapsing `σ_min` means the iterate is
walking off the edge of the fixed-rank manifold, where the Hessian is not defined at all and "the
probe was inconclusive" and "there is no interior minimum" stop being distinguishable. That is not
a failure — it is the optimizer saying the rank is too *high*. Log it at every stage of a descent,
and trim before trusting [`rpa`](@ref) or [`pd_probe`](@ref).
"""
sigma_min(x::TTVector{T,L}) where {T,L} =
    minimum(minimum(b) for b in schmidt(x).sigma if !isempty(b))

"""
    grad_norm(H, x) -> Float64

`‖P_T(Hx − λx)‖`, the Riemannian gradient norm at `x` with `λ` its Rayleigh quotient — how
critical the state is. The RPA reference criterion is `< 1e-9`: positive-definiteness after radial
deflation holds only *at* the minimizer, so a PD verdict at a non-critical point is not a statement
about a minimizer.

This is the gradient at `x` as given, so it is the meaningful number for a normalized state
(which every state a solver returns here is) and scales with an unnormalized one.
"""
grad_norm(h::Hamiltonian{T,L}, x::TTVector{T,L}) where {T,L} =
    _grad_norm(L, T, _ptr(h), _ptr(x))

# =================================================================================================
# Persistence
# =================================================================================================

"""
    save(x, path)

Write the tensor train to `path` in the library's `QNTT-TT` v1 format, truncating.

**This is a cache, not an archive.** The format version identifies what the loaded shared library
writes; there is no forward-compatibility promise and no migration path. If [`load`](@ref) refuses
a file, delete it and recompute rather than trying to convert it.

The motivating use is skipping a ground-state re-derivation: the pipeline is deterministic, so
re-running MALS + polish + trim to reach a state that is already on disk is pure waste. A cache
whose space, rank grid or source fixture disagrees with the run you are doing is worse than no
cache — check those after loading; the format validates its own integrity, not your intent.

Point-group symmetry is carried in the file, so a `|G| > 1` train round-trips exactly.
"""
save(x::TTVector{T,L}, path::AbstractString) where {T,L} =
    (_vec_save(L, T, _ptr(x), String(path)); nothing)

"""
    load(T, L, path) -> TTVector{T,L}
    load(TTVector{T,L}, path) -> TTVector{T,L}
    load(path) -> TTVector{Float64,Abelian}

Read a tensor train written by [`save`](@ref).

The file records its own layer and scalar and is checked against the ones you ask for, so loading
an SU(2) state as abelian (or an `Float64` state as `ComplexF64`) throws rather than
reinterpreting bytes. So does an unknown format version, a truncated file, or a rank grid that
disagrees with the coefficient stream.

```julia
x = QNTT.load(Float64, QNTT.SU2, "h2o_gs.tt")
y = QNTT.load(QNTT.TTVector{ComplexF64,QNTT.Abelian}, "seed.tt")
```
"""
load(::Type{T}, ::Type{L}, path::AbstractString) where {T,L<:Layer} =
    TTVector{T,L}(_vec_load(L, T, String(path)))
load(::Type{TTVector{T,L}}, path::AbstractString) where {T,L<:Layer} = load(T, L, path)
load(path::AbstractString) = load(Float64, Abelian, path)

